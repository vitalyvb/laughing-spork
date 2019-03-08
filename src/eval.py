#!/usr/bin/env python3

from functools import reduce
import time
import unittest
from collections import deque

from etypes import *

def ENil():
    return Nil(0, 0)

def ESym(v):
    return Sym(0, 0, v)

def ENum(v):
    return Num(0, 0, v)

def EList(v):
    return EvalList(0, 0, v)

def VList(v):
    return List(0, 0, v)

def ELambda(v, e):
    return Lambda(0, 0, v, e)

class EvalRuntimeError(Exception):
    pass

class EvalQuoted(object):
    start = None
    end = None

Rest = type("Rest", (EvalQuoted,), {})()
Begin = type("Begin", (EvalQuoted,), {})()
Call = type("Call", (EvalQuoted,), {})()
CallCC = type("CallCC", (EvalQuoted,), {})()
Eval = type("Eval", (EvalQuoted,), {})()
Quote = type("Quote", (EvalQuoted,), {})()
Define = type("Define", (EvalQuoted,), {})()
EIf = type("EIf", (EvalQuoted,), {})()
DefMacro = type("DefMacro", (EvalQuoted,), {})()
CallMacro = type("CallMacro", (EvalQuoted,), {})()

EIf.format = lambda align: align("EIf")

class Macro(object):
    def __init__(self, exp):
        self.exp = exp


class Frame(object):
    def __init__(self, env, data, cont, descr="", src=None):
        self.descr = ""
        self.env = env
        self.data = data
        self.cont = cont

        if src and hasattr(src, "start") and src.start:
            self.descr += "{:3d}:{:2d} ".format(src.start.line, src.start.column)
        else:
            self.descr += "       "
        if descr:
            self.descr += descr

    def __repr__(self):
        return repr(self.data)


class Stats(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self._max_recurs = 0
        self._max_stack = 0
        self._max_env = 0

    def recurs(self, v):
        if v > self._max_recurs:
            self._max_recurs = v

    def envd(self, e):
        if e._cnt > self._max_env:
            self._max_env = e._cnt

    def stack(self, v):
        if v > self._max_stack:
            self._max_stack = v

    def print(self):
        print("---------------")
        print("VM stats:")
        print("    Max Python recursion: {}".format(self._max_recurs))
        print("    Max VM stack length:  {}".format(self._max_stack))
        print("    Max VM ENV length:    {}".format(self._max_env))


class VMEval(object):


    def __init__(self, trace=0):
        self.trace_size = trace
        self.stats = Stats()
        self.clear()

    def clear(self):
        self.stats.clear()
        self.traces = deque(maxlen=self.trace_size)
        self.recurs = 0

    def add_trace_point(self, exp, env):

        if self.trace_size <= 0:
            return

        if hasattr(exp, 'start') and exp.start:
                l = exp.start.buffer.splitlines()
                ln = exp.start.line
                cn = exp.start.column

                if isinstance(exp, Sym) and exp.v in env and isinstance(env[exp.v], (Num,Str)):
                    val = "--  {} = {}".format(exp.v, env[exp.v].v)
                else:
                    val = ""

                txt = []
                if ln-1 >= 0:
                    txt.append("{:3}: {}".format(ln-1, l[ln-1]))
                txt.append("{:3}: {}".format(ln, l[ln]))
                txt.append("     "+" "*cn+"^"+val)
                if ln+1 < len(l):
                    txt.append("{:3}: {}".format(ln+1, l[ln+1]))

                self.traces.append(txt)

#                print(exp.start)
#                time.sleep(0.3)
        else:
            self.traces.append([str(exp)])

    def dump_trace(self):
        pos = len(self.traces)
        print("Dumping trace:")
        while self.traces:
            print("    {}:".format(pos))
            txt = self.traces.popleft()
            print("\n".join("        "+l for l in txt))
            pos -= 1

    def __call__(self, env, exp):
        self.recurs += 1

        self.stats.recurs(self.recurs)

        if self.recurs > 5:
            print("eval1 recursion level: {}".format(self.recurs))

        if isinstance(exp, Module):
            exp = exp.v

        stack = deque()

        try:
            if isinstance(exp, list):
                res = []
                for e in exp:
                    v = eval2(self, stack, env.copy(), e)
                    if not isinstance(v, Nil):
                        res.append(v)
#                    print("#######################", v)
                return res

            return eval2(self, stack, env, exp)
        except:

            if self.trace_size > 0:
                self.dump_trace()
#            print("====================")
#            while stack:
#                frame = stack.pop()
#                print(frame)
#            print("====================")
            raise
        finally:
            self.recurs -= 1

eval1 = VMEval(trace=10)

def eval2(vm, stack, env, exp):

    def s_push(env, cont, data, descr=None, src=None):
        stack.append(Frame(env, data, cont, descr=descr, src=src))

    def s_reset(st):
        while stack:
            stack.pop()
        stack.extend(st)

    def s_pop():
        frame = stack.pop()
        return frame


    def op_list(env, exp):
        if len(exp.v) == 0:
            return (VList([]),)
        s_push(env, op_list2, (exp, []), descr="List:1", src=exp)
        return exp.v[0]

    def op_list2(env, exp, _cexp, cdata, frame):
        cdata, acc = cdata
        acc = acc + [exp]
        if len(acc) >= len(cdata.v):
            return (VList(acc),)

        s_push(env, op_list2, (cdata, acc), descr="List:N", src=exp)
        return cdata.v[len(acc)]


    def op_pylist(env, exp):
        if len(exp) > 1:
            s_push(env, op_pylist2, (exp, []), descr="pylist:1", src=exp)

#        env.weaken()
        return exp[0]

    def op_pylist2(env, exp, _cexp, cdata, frame):
        cdata, acc = cdata
        acc = acc + [exp]

        if len(acc) >= len(cdata)-1:
#            env.weaken()
            return cdata[len(acc)]

        s_push(env, op_pylist2, (cdata, acc), descr="pylist:N", src=exp)
        return cdata[len(acc)]

    def callcc(envn):
        res = envn["&cont"]
        (stack0, env0) = envn["&closure"]

        s_reset(stack0)
#        envn.reset(env0)

        return (res,)

    def op_eval(env, f, _cexp, _cdata, frame):
        if isinstance(f, (List, EvalList)):
            f = f.v
        return f

    def op_define(env, f, _cexp, cdata, frame):
        env.set(cdata.v, f)
        return (ENil(),)

    def op_macro(env, f, _cexp, cdata, frame):
        env.set("!"+cdata.v, Macro(f))
        return (ENil(),)

    def op_callmacro(env, f, _cexp, cdata, frame):
        return Apply(0,0, env["!"+cdata.v].exp, list(Apply(0,0, Quote, [x]) for x in f.v))

    def op_eif2(env, f, _cexp, cdata, frame):
        if not isinstance(f, Nil):
            exp = cdata[0]
        elif len(cdata) > 1:
            exp = cdata[1]
        else:
            return ENil()
        env.weaken()
        return Apply(0,0, exp, [])

    def op_apply(env, exp):
        s_push(env, op_apply2, exp, descr="apply:E", src=exp)
        return exp.sym

    def op_apply2(env, f, _cexp, cdata, frame):

        if f is Begin:
            return cdata.args

        if f is EIf:
            if len(cdata.args) not in (2, 3):
                raise EvalRuntimeError("&if requires 2 or 3 arguments")
            s_push(env, op_eif2, cdata.args[1:], descr="apply:if", src=cdata)
            return cdata.args[0]

        if f is Define:
            if len(cdata.args) < 2:
                raise EvalRuntimeError("define requires 2 arguments or more")
            s_push(env, op_define, cdata.args[0], descr="apply:def", src=cdata)
            return cdata.args[1:]

        if f is DefMacro:
            if len(cdata.args) < 2:
                raise EvalRuntimeError("macro requires 2 arguments - symbol and lambda")
            s_push(env, op_macro, cdata.args[0], descr="apply:mac", src=cdata)
            return cdata.args[1:]

        if f is CallMacro:
            if len(cdata.args) != 2:
                raise EvalRuntimeError("callmacro requires 2 arguments - symbol and args list")

            s_push(env, op_callmacro, cdata.args[0], descr="apply:mc", src=cdata)
            return cdata.args[1]

        if f is Eval:
            if len(cdata.args) != 1:
                raise EvalRuntimeError("eval requires 1 argument of list()")

            s_push(env, op_eval, None, descr="apply:eval", src=cdata)
            return cdata.args

        if f is Quote:
            if len(cdata.args) < 1:
                raise EvalRuntimeError("quote requires at least 1 argument")

            if len(cdata.args) == 1:
                return (cdata.args[0],)

            return (VList(cdata.args),)

        if f is CallCC:
            if len(cdata.args) != 1:
                raise EvalRuntimeError("call/cc requires 1 argument of lambda(x)")

            stack0 = deque(stack)
            env0 = env.copy()
            cc = Closure((stack0, env0), VList([ESym("&cont")]), callcc)
            # whatever cdata.args[0] is, Apply will evaluate it
            ret = Apply(0,0, cdata.args[0], [cc]);
            return ret

        if isinstance(f, Macro):
            return Apply(0,0, Eval, [Apply(0,0, f.exp, list(Apply(0,0, Quote, [x]) for x in cdata.args))])

        for n, x in enumerate(f.params.v):
            if x.v == "&rest":
                p = n
                args = f.params.v[:p]
                argv = f.params.v[p+1]
                break
        else:
            args = f.params.v
            argv = None

        ctx = (f, args, argv)
        largs = cdata.args

        s_push(env, op_apply3, ctx, descr="apply:A", src=cdata)
        if isinstance(largs, list):
            return EList(largs)

        if isinstance(largs, Sym):
            return EList([largs])

        raise Exception("unexpected arguments type '{}' when calling '{}'".format(largs, f))

    def op_apply3(env, exp, _cexp, cdata, frame):
        (f, args, argv) = cdata

        if len(args) > len(exp.v):
            raise Exception("not enough arguments to call {}".format(f))

        if argv is None and len(args) < len(exp.v):
            raise Exception("too many arguments to call {}".format(f))

        alls = list(args)
        if argv is not None:
            alls.append(argv)

        if isinstance(f, Closure):
            env = f.env[1]

        env = Env(prev=env, local=[x.v for x in alls])
        frame.env = env

        for arg, val in zip(args, exp.v):
            env[arg.v] = val

        if argv is not None:
            env[argv.v] = VList(exp.v[len(args):])

        if isinstance(f, Closure):
            env["&closure"] = (f.env[0], env)

        return f.exp

    def op_callable(env, exp):
        return exp(env)


    while True:

        while True:

#            print(exp)

            if len(stack) > 100:
                print("my stack size: {}".format(len(stack)))
                for x in stack:
                    print("{:12s} - {}".format(x.descr, x))
                1/0
            vm.stats.stack(len(stack))
            vm.stats.envd(env)

            vm.add_trace_point(exp, env)


            if isinstance(exp, Sym):
                if "!"+exp.v in env:
                    exp = env["!"+exp.v]
                else:
                    exp = env[exp.v]
                break

            if isinstance(exp, (Nil, Num, Str, Closure, EvalQuoted, List)):
                break

            if isinstance(exp, Apply):
                exp = op_apply(env, exp)
                continue

            if isinstance(exp, EvalList):
                exp = op_list(env, exp)
                continue

            if isinstance(exp, type((0,))):
                exp = exp[0]
                break

            if callable(exp):
                exp = op_callable(env, exp)
                continue

            if isinstance(exp, list) and exp == []:
                exp = ENil()
                break

            if isinstance(exp, list):
                exp = op_pylist(env, exp)
                continue

            if isinstance(exp, Lambda):
                env0 = env.copy()
                exp = Closure((None, env0), exp.params, exp.exp)
                break


            raise Exception("Eval '{}' not implemented".format(type(exp)))

        if len(stack) == 0:
            break

        frame = s_pop()
        exp = frame.cont(frame.env, exp, None, frame.data, frame)
        env = frame.env

    return exp


class Env(object):
    def __init__(self, prev=None, local=None, cnt=1):
        self._weak = False

        if prev is None:
            self._globals = {}
            self._init()
        else:
            self._globals = prev._globals
            cnt = prev._cnt + 1

            if prev._weak:
                prev = prev._prev_env

        self._prev_env = prev

        self._locals = {}
        if local is not None:
            for k in local:
                self._locals[k] = None

        self._cnt = cnt

#    def copy(self, local=None):
#        return Env(prev=self, local=local)

    def copy(self):
        e = Env()
        e._globals = self._globals
        e._locals = self._locals.copy()
        e._prev_env = self._prev_env.copy() if self._prev_env is not None else None
        return e

    def _init(self):
        self._globals["&rest"] = Rest
        self._globals["begin"] = Begin
        self._globals["call"] = Call
        self._globals["call/cc"] = CallCC
        self._globals["eval"] = Eval
        self._globals["quote"] = Quote
        self._globals["define"] = Define
        self._globals["defmacro"] = DefMacro
        self._globals["callmacro"] = CallMacro
        self._globals["&if"] = EIf

    #def reset(self, other):
    #    self._globals = other._globals
    #    self._locals = other._locals
    #    self._prev_env = other._prev_env

    def weaken(self):
        self._weak = True

    def search(self, s, default=None):
        # search in all locals, then globals
        if s in self._locals:
            # this locals
            return self._locals

        if self._prev_env is not None:
            # prev env's locals, or globals if none
            return self._prev_env.search(s, default)

        if s in self._globals:
            return self._globals

        return default

    def set(self, s, v):
        default = self._globals
        d = self.search(s, default)
        d[s] = v

    def get(self, s):
        d = self.search(s)
        if d is not None:
            return d[s]
        else:
            raise Exception("symbol '{}' not defined".format(s))

    def __contains__(self, i):
        d = self.search(i)
        return d is not None and i in d

    def __getitem__(self, i):
        return self.get(i)

    def __setitem__(self, i, v):
        return self.set(i, v)

    #def __delitem__(self, i):
    #    del self._globals[i]

    def add_builtin(self, tr, s, ar, func=None, rawfunc=None):
        args = []
        argv = None
        for i, v in enumerate(ar):
            if v == "&rest":
                break
            args.append(v)

        if v == "&rest":
            argv = ar[i+1]

        if argv is not None:
            args.append(argv)

        if func is not None:
            def value(env):
                fargs = list(map(lambda v: env[v], args))
                return (tr(func(*tuple(fargs))),)
            val = ELambda(VList(list(map(ESym, ar))), value)
        elif rawfunc is not None:
            val = rawfunc
        self._globals[s] = val


def get_prelude_env():
    new_env = Env()

    define = new_env.add_builtin

    _id = lambda x:x

    define(_id, "debug", ["&rest", "args"], lambda args: (ENil(), print(args.v))[0] )
    define(_id, "display", ["&rest", "args"], lambda args: (ENil(), print(" ".join(x.v for x in args.v), end=""))[0] )
    define(_id, "sleep", ["&rest", "args"], lambda args: (ENil(), time.sleep(0.5))[0] )

    define(ENum, "+", ["&rest", "args"], lambda args: reduce(lambda a,b: a+b.v, args.v, 0) )
    define(ENum, "*", ["&rest", "args"], lambda args: reduce(lambda a,b: a*b.v, args.v, 1) )
    define(ENum, "-", ["i", "&rest", "args"], lambda i, args: reduce(lambda a,b: a-b.v, args.v, i.v) )

    def unzero(n):
        if n == 0:
            raise EvalRuntimeError("division by zero")
        return n
    define(ENum, "/", ["i", "&rest", "args"], lambda i, args: reduce(lambda a,b: a//unzero(b.v), args.v, i.v) )

    define(_id, "eq?", ["a", "b"], lambda a, b: [ENil(), VList([])][a.v == b.v] )

    define(_id, "list", ["&rest", "args"], lambda args: args )

    def unempty(n):
        if len(n) == 0:
            raise EvalRuntimeError("empty list")
        return n
    define(_id, "first", ["lst"], lambda lst: unempty(lst.v)[0])
    define(_id, "rest", ["lst"], lambda lst: VList(unempty(lst.v)[1:]))

    def _cons(x, xs):
        if isinstance(xs, Nil):
            return VList([x])
        return VList([x]+xs.v)
    define(_id, "cons", ["x", "xs"], _cons)

    define(_id, "list?", ["x"], lambda x: [ENil(), VList([])][isinstance(x, List)] )

    define(ENum, "length", ["lst"], lambda lst: len(lst.v) )
    define(_id, "empty?", ["lst"], lambda lst: [ENil(), VList([])][isinstance(lst, List) and len(lst.v) == 0] )

    define(_id, "elem", ["lst", "idx"], lambda lst, idx: lst.v[idx.v] )

    define(_id, "&apply", ["exp", "args"], lambda exp, args: Apply(0,0, exp, args.v) )
    define(_id, "&apply?", ["app"], lambda app: [ENil(), VList([])][isinstance(app, Apply)] )
    define(_id, "&apply.exp", ["app"], lambda app: app.sym )
    define(_id, "&apply.args", ["app"], lambda app: VList(app.args) )
    define(_id, "&apply.tolist", ["app"], lambda app: VList([app.sym]+app.args) )

    define(_id, "&list.elist", ["lst"], lambda lst: EList(lst.v) )
    define(_id, "&list.eval", ["lst"], lambda lst: lst.v )

    define(_id, "&lambda", ["args", "exp"], lambda args, exp: Lambda(0,0, args, exp) )

    define(_id, "&format", ["exp"], lambda exp: (ENil(), print(exp.format()))[0] )

    def eif(env):
        if not isinstance(env["exp"], Nil):
            exp = env["fthen"]
        else:
            felse = env["felse"].v
            if len(felse) == 1:
                exp = felse[0]
            elif len(felse) > 1:
                raise EvalRuntimeError("too may elses in if")
            else:
                return ENil()
#        env.weaken()
        return exp.exp
#        return Apply(0,0, exp, [])

    define(_id, "&if", ["exp", "fthen", "&rest", "felse"], rawfunc=ELambda(VList(list(map(ESym, ["exp", "fthen", "&rest", "felse"]))),eif) )

    return new_env


def _lmb_If(x):
    return Lambda(0,0, VList([]), x)

def If(_s, _e, x, t, e=None):
    if e is None:
        return Apply(0,0, ESym("&if"), [x, _lmb_If(t)])
    return Apply(0,0, ESym("&if"), [x, _lmb_If(t), _lmb_If(e)])


class Test_Eval(unittest.TestCase):

    def setUp(self):
        self.env = get_prelude_env()

    @classmethod
    def tearDownClass(cls):
        eval1.stats.print()

    def test_unhandled1_raises(self):
        class _Unhandled(object):
            pass
        p = _Unhandled()
        with self.assertRaises(Exception):
            eval1(self.env, p)


    def test_begin1(self):
        p = Apply(0,0, ESym("begin"), [
                ENum(21),
                ENum(42)
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 42)

    def test_begin2(self):
        p = Apply(0,0, ESym("begin"), [])
        res = eval1(self.env, p)
        self.assertIsInstance(res, Nil)


    def test_define(self):
        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("a"), ENum(12)),
                Def(0,0, ESym("a"), ENum(42)),
                ESym("a")
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 42)

    def test_define2(self):
        p = Apply(0,0, ESym("begin"), [
                Apply(0,0, ESym("define"), [ESym("a"), ENum(12)] ),
                Apply(0,0, ESym("define"), [ESym("a"), ENum(21), ENum(42)] ),
                ESym("a")
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 42)


    def test_multiply1(self):
        p = Apply(0,0, ESym("*"), [])
        res = eval1(self.env, p)
        self.assertEqual(res, 1)

    def test_multiply2(self):
        p = Apply(0,0, ESym("*"), [ENum(44)])
        res = eval1(self.env, p)
        self.assertEqual(res, 44)

    def test_multiply3(self):
        p = Apply(0,0, ESym("*"), [ENum(3), ENum(5), ENum(11)])
        res = eval1(self.env, p)
        self.assertEqual(res, 165)


    def test_add1(self):
        p = Apply(0,0, ESym("+"), [])
        res = eval1(self.env, p)
        self.assertEqual(res, 0)

    def test_add2(self):
        p = Apply(0,0, ESym("+"), [ENum(44)])
        res = eval1(self.env, p)
        self.assertEqual(res, 44)

    def test_add3(self):
        p = Apply(0,0, ESym("+"), [ENum(3), ENum(5), ENum(11)])
        res = eval1(self.env, p)
        self.assertEqual(res, 19)


    def test_sub1_raises(self):
        p = Apply(0,0, ESym("-"), [])
        with self.assertRaises(Exception):
            res = eval1(self.env, p)

    def test_sub2(self):
        p = Apply(0,0, ESym("-"), [ENum(44)])
        res = eval1(self.env, p)
        self.assertEqual(res, 44)

    def test_sub3(self):
        p = Apply(0,0, ESym("-"), [ENum(3), ENum(5), ENum(11)])
        res = eval1(self.env, p)
        self.assertEqual(res, -13)


    def test_div1_raises(self):
        p = Apply(0,0, ESym("/"), [])
        with self.assertRaises(Exception):
            res = eval1(self.env, p)

    def test_div2(self):
        p = Apply(0,0, ESym("/"), [ENum(44)])
        res = eval1(self.env, p)
        self.assertEqual(res, 44)

    def test_div3(self):
        p = Apply(0,0, ESym("/"), [ENum(50), ENum(5), ENum(2)])
        res = eval1(self.env, p)
        self.assertEqual(res, 5)

    def test_div4(self):
        p = Apply(0,0, ESym("/"), [ENum(1), ENum(2)])
        res = eval1(self.env, p)
        self.assertEqual(res, 0)

    def test_div5_raises(self):
        p = Apply(0,0, ESym("/"), [ENum(1), ENum(0)])
        with self.assertRaises(EvalRuntimeError):
            eval1(self.env, p)


    def test_eq1_raises(self):
        p = Apply(0,0, ESym("eq?"), [])
        with self.assertRaises(Exception):
            eval1(self.env, p)

    def test_eq2_raises(self):
        p = Apply(0,0, ESym("eq?"), [ENum(44)])
        with self.assertRaises(Exception):
            eval1(self.env, p)

    def test_eq3(self):
        p = Apply(0,0, ESym("eq?"), [ENum(3), ENum(5)])
        res = eval1(self.env, p)
        self.assertIsInstance(res, Nil)

    def test_eq4(self):
        p = Apply(0,0, ESym("eq?"), [ENum(3), ENum(3)])
        res = eval1(self.env, p)
        self.assertNotIsInstance(res, Nil)


    def test_if1(self):
        p = If(0,0, ENil(), ENum(10), ENum(20))
        res = eval1(self.env, p)
        self.assertEqual(res, 20)

    def test_if2(self):
        p = If(0,0, EList([]), ENum(10), ENum(20))
        res = eval1(self.env, p)
        self.assertEqual(res, 10)

    def test_if3(self):
        p = If(0,0, ENil(), ENum(10), ENil())
        res = eval1(self.env, p)
        self.assertIsInstance(res, Nil)

    def test_if4(self):
        p = If(0,0, EList([]), ENum(10), ENil())
        res = eval1(self.env, p)
        self.assertEqual(res, 10)

    def test_if5(self):
        p = If(0,0, EList([]), ENum(10))
        res = eval1(self.env, p)
        self.assertEqual(res, 10)

    def test_if6(self):
        p = If(0,0, ENil(), ENum(10))
        res = eval1(self.env, p)
        self.assertIsInstance(res, Nil)


    def test_list1(self):
        p = Apply(0,0, ESym("list"), [])
        res = eval1(self.env, p)
        self.assertEqual(res, VList([]))

    def test_list2(self):
        p = Apply(0,0, ESym("list"), [ENum(1), ENum(2), ENum(3)])
        res = eval1(self.env, p)
        self.assertEqual(res, VList([ENum(1), ENum(2), ENum(3)]))

    def test_list3(self):
        p = [
                Def(0,0, ESym("xs"), Apply(0,0, ESym("list"), [ENum(1), ENum(2), ENum(3)])),
                Apply(0,0, ESym("first"), ESym("xs")),
                Apply(0,0, ESym("rest"), ESym("xs")),
                Apply(0,0, ESym("first"), ESym("xs")),
            ]
        res = eval1(self.env, p)
        self.assertEqual(res, [ENum(1), VList([ENum(2), ENum(3)]), ENum(1)])

    def test_list4_raises(self):
        p = [
                Def(0,0, ESym("xs"), Apply(0,0, ESym("list"), [])),
                Apply(0,0, ESym("first"), ESym("xs")),
            ]
        with self.assertRaises(EvalRuntimeError):
            eval1(self.env, p)

    def test_list5_raises(self):
        p = [
                Def(0,0, ESym("xs"), Apply(0,0, ESym("list"), [])),
                Apply(0,0, ESym("rest"), ESym("xs")),
            ]
        with self.assertRaises(EvalRuntimeError):
            eval1(self.env, p)

    def test_list6_cons(self):
        p = [
                Def(0,0, ESym("xs"), Apply(0,0, ESym("list"), [])),
                Apply(0,0, ESym("cons"), [ENum(42),  ESym("xs")]),
                Apply(0,0, ESym("cons"), [ENum(42),  ENil()]),

                Def(0,0, ESym("xs2"), Apply(0,0, ESym("list"), [ENum(1), ENum(2)])),
                Apply(0,0, ESym("cons"), [ENum(42),  ESym("xs2")]),

            ]
        res = eval1(self.env, p)
        self.assertEqual(res, [VList([ENum(42)]),
                               VList([ENum(42)]),
                               VList([ENum(42), ENum(1), ENum(2)]) ])

    def test_list7(self):
        p = [ EList ([
                Apply(0,0, ESym("list?"), [VList([])]),
                Apply(0,0, ESym("list?"), [ENil()]),
                Apply(0,0, ESym("list?"), [VList([ENum(11)])]),
                Apply(0,0, ESym("list?"), [ENum(22)]),
                Apply(0,0, ESym("list?"), [EList([])]),
            ])]
        res = eval1(self.env, p)
        self.assertEqual(len(res[0].v), 5)
        self.assertEqual(res[0].v[0], VList([]))
        self.assertIsInstance(res[0].v[1], Nil)
        self.assertEqual(res[0].v[2], VList([]))
        self.assertIsInstance(res[0].v[3], Nil)
        self.assertEqual(res[0].v[4], VList([]))

    def test_list8(self):
        p = [ EList ([
                Apply(0,0, ESym("length"), [VList([])]),
                Apply(0,0, ESym("length"), [VList([1])]),
                Apply(0,0, ESym("length"), [VList([1,2])]),
            ])]
        res = eval1(self.env, p)
        self.assertEqual(len(res[0].v), 3)
        self.assertEqual(res[0].v[0], ENum(0))
        self.assertEqual(res[0].v[1], ENum(1))
        self.assertEqual(res[0].v[2], ENum(2))

    def test_list9(self):
        p = [ EList ([
                Apply(0,0, ESym("elem"), [VList([ENum(1),ENum(2),ENum(3)]), ENum(1)]),
            ])]
        res = eval1(self.env, p)
        self.assertEqual(len(res[0].v), 1)
        self.assertEqual(res[0].v[0], ENum(2))


    def test_lambda1(self):
        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([]), ENum(42))),
                Def(0,0, ESym("a"), ENum(11)),
                Apply(0,0, ESym("f"), [])
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 42)

    def test_lambda2(self):
        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([ESym("x")]), ESym("x"))),
                Apply(0,0, ESym("f"), [ENum(33)])
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 33)

    def test_lambda3_raises(self):
        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([ESym("x")]), ESym("x"))),
                Apply(0,0, ESym("f"), [])
            ])

        with self.assertRaises(Exception):
            eval1(self.env, p)

    def test_lambda4_raises(self):
        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([ESym("x")]), ESym("x"))),
                Apply(0,0, ESym("f"), [ENum(33), ENum(44)])
            ])

        with self.assertRaises(Exception):
            eval1(self.env, p)

    def test_lambda5(self):
        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([ESym("x"), ESym("&rest"), ESym("xs"),]), ESym("xs"))),
                Apply(0,0, ESym("f"), [ENum(33), ENum(44), ENum(55)])
            ])

        res = eval1(self.env, p)
        self.assertIsInstance(res, List)
        self.assertEqual(res.v, [ENum(44),ENum(55)])

    def test_lambda6(self):
        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([ESym("x"), ESym("&rest"), ESym("xs"),]), ESym("xs"))),
                Apply(0,0, ESym("f"), [ENum(33)])
            ])

        res = eval1(self.env, p)
        self.assertIsInstance(res, List)
        self.assertEqual(res.v, [])

    def test_lambda7_raises(self):
        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([ESym("x"), ESym("&rest"), ESym("xs"),]), ESym("xs"))),
                Apply(0,0, ESym("f"), [])
            ])

        with self.assertRaises(Exception):
            eval1(self.env, p)

    def test_lambda8(self):
        def add(s, n):
            return Apply(0,0, ESym("+"), [ESym(s), ENum(n)])

        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([ESym("x")]), ESym("x"))),
                Def(0,0, ESym("g"), ELambda(EList([ESym("x")]), add("x", 2) )),
                Def(0,0, ESym("h"), Apply(0,0, ESym("f"), [ESym("g")] )),
                Apply(0,0, ESym("h"), [ENum(2)])
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 4)

    def test_lambda9(self):
        def add(s, n):
            return Apply(0,0, ESym("+"), [ESym(s), ENum(n)])

        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([ESym("x")]), add("x", 3))),
                Def(0,0, ESym("g"), ELambda(EList([ESym("x")]), add("x", 2))),
                Def(0,0, ESym("h"), ELambda(EList([ESym("x")]), Apply(0,0, ESym("f"), [Apply(0,0, ESym("g"), [ESym("x")] )] ))),
                Apply(0,0, ESym("h"), [ENum(10)])
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 15)

    def test_lambda10(self):
        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([ESym("x"), ESym("y")]), [ESym("x"), ESym("y")])),
                Apply(0,0, ESym("f"), [ENum(11), ENum(22)])
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 22)


    def test_tail_call(self):
        def add(s, n):
            return Apply(0,0, ESym("+"), [ESym(s), n])

        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([ESym("acc"), ESym("x")]),
                    If(0,0, Apply(0,0, ESym("eq?"), [ESym("x"), ENum(0)]),
                            ESym("acc"),
                            Apply(0,0, ESym("f"), [add("acc", ESym("x")),
                                                   add("x", ENum(-1))])
                    ))),
                Apply(0,0, ESym("f"), [ENum(0), ENum(1000)])
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 500500)

    def test_tail_call2(self):
        def add(s, n):
            return Apply(0,0, ESym("+"), [ESym(s), n])

        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([ESym("acc"), ESym("x")]),
                    Apply(0,0, ESym("begin"), [
                        If(0,0, Apply(0,0, ESym("eq?"), [ESym("x"), ENum(0)]),
                                ESym("acc"),
                                Apply(0,0, ESym("f"), [add("acc", ESym("x")),
                                                   add("x", ENum(-1))]))
                    ]))),
                Apply(0,0, ESym("f"), [ENum(0), ENum(1000)])
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 500500)

    def test_tail_call3(self):
        def add(s, n):
            return Apply(0,0, ESym("+"), [ESym(s), n])

        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("is-even?"), ELambda(EList([ESym("x")]),
                        If(0,0, Apply(0,0, ESym("eq?"), [ESym("x"), ENum(0)]),
                                EList([]),
                                Apply(0,0, ESym("is-odd?"), [add("x", ENum(-1))])))),

                Def(0,0, ESym("is-odd?"), ELambda(EList([ESym("x")]),
                        If(0,0, Apply(0,0, ESym("eq?"), [ESym("x"), ENum(0)]),
                                ENil(),
                                Apply(0,0, ESym("is-even?"), [add("x", ENum(-1))])))),

                EList([
                    Apply(0,0, ESym("is-even?"), [ENum(1000)]),
                    Apply(0,0, ESym("is-even?"), [ENum(1001)]),
                ])
            ])

        res = eval1(self.env, p)
        self.assertIsInstance(res, List)
        self.assertEqual(res.v[0], VList([]))
        self.assertIsInstance(res.v[1], Nil)


    def test_closure1(self):
        def add(s, n):
            return Apply(0,0, ESym("+"), [ESym(s), ENum(n)])

        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([ESym("x")]), add("x", 7))),
                Def(0,0, ESym("g"), ELambda(EList([ESym("x")]), add("x", 5))),

                Def(0,0, ESym("h"), ELambda(EList([ESym("x"), ESym("y")]),
                                        ELambda(EList([ESym("z")]),
                                            Apply(0,0, ESym("x"), [Apply(0,0, ESym("y"), [ESym("z")] )] )))),

                Def(0,0, ESym("q"), Apply(0,0, ESym("h"), [ESym("f"), ESym("g")])),
                Apply(0,0, ESym("q"), [ENum(10)])
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 22)

#    @unittest.skip("bug")
    def test_closure2(self):
        def add(s, n):
            return Apply(0,0, ESym("+"), [ESym(s), ENum(n)])

        p = Apply(0,0, ESym("begin"), [

                Def(0,0, ESym("f"), ELambda(EList([ESym("x")]), add("x", 3))),
                Def(0,0, ESym("g"), ELambda(EList([ESym("x")]), add("x", 2))),

                Def(0,0, ESym("h"), ELambda(EList([ESym("f"), ESym("g")]),
                                        ELambda(EList([ESym("x")]),
                                            Apply(0,0, ESym("g"), [Apply(0,0, ESym("f"), [ESym("x")] )] )))),

                Def(0,0, ESym("f"), ELambda(EList([ESym("x")]), add("x", 7))),
                Def(0,0, ESym("g"), ELambda(EList([ESym("x")]), add("x", 5))),
                Def(0,0, ESym("q"), Apply(0,0, ESym("h"), [ESym("f"), ESym("g")])),
                Apply(0,0, ESym("q"), [ENum(10)])
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 22)

    def test_closure3(self):

        p = [
            Def(0,0, ESym("counter"),
                Apply(0,0,
                    ELambda(EList([ESym("state")]),
                        ELambda(EList([]),
                            [Def(0,0, ESym("state"), Apply(0,0, ESym("+"), [ESym("state"), ENum(1)])),
                            ESym("state")]
                        )
                    ),
                    [ENum(0)]),
            ),

            Def(0,0, ESym("state"), ENum(33)),

            Apply(0,0, ESym("counter"), []),
            Apply(0,0, ESym("counter"), []),

            Def(0,0, ESym("state"), ENum(22)),
            Apply(0,0, ESym("counter"), []),
            Apply(0,0, ESym("counter"), []),

        ]

        res = eval1(self.env, p)
        self.assertEqual(res, [1,2,3,4])

    def test_scope_dyn1(self):
        def add(s, n):
            return Apply(0,0, ESym("+"), [ESym(s), ENum(n)])

        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("a"), ENum(12)),
                Def(0,0, ESym("f"), ELambda(EList([]), add("a", 3))),
                Def(0,0, ESym("a"), ENum(42)),

                Apply(0,0, ESym("f"), []),
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 45)

    def test_scope_dyn2(self):
        def add(s, n):
            return Apply(0,0, ESym("+"), [ESym(s), ENum(n)])

        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("a"), ENum(12)),
                Def(0,0, ESym("f"), ELambda(EList([]), add("a", 3))),
                Def(0,0, ESym("r"), Apply(0,0, ESym("f"), [])),
                Def(0,0, ESym("a"), ENum(42)),

                ESym("r"),
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 15)

    @unittest.skip("bug")
    def test_scope_dyn3(self):
        def add(s, n):
            return Apply(0,0, ESym("+"), [ESym(s), ENum(n)])

        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("a"), ENum(42)),
                Def(0,0, ESym("f"), ELambda(EList([ESym("a")]), [Def(0,0, ESym("a"), ENum(3)), ESym("a")])),
                Def(0,0, ESym("r"), Apply(0,0, ESym("f"), [ENum(33)])),

                EList([ESym("r"), ESym("a")])
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, VList([3, 42]))

    @unittest.skip("bug")
    def test_scope_dyn3a(self):
        def add(s, n):
            return Apply(0,0, ESym("+"), [ESym(s), ENum(n)])

        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("a"), ENum(42)),
                Def(0,0, ESym("f"), ELambda(EList([ESym("a")]), ESym("a"))),
                Def(0,0, ESym("r"), Apply(0,0, ESym("f"), [ENum(33)])),

                EList([ESym("r"), ESym("a")])
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, VList([33, 42]))

    def test_scope_dyn4(self):
        def add(s, n):
            return Apply(0,0, ESym("+"), [ESym(s), ENum(n)])

        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("a"), ENum(42)),
                Def(0,0, ESym("f"), ELambda(EList([ESym("b")]), [Def(0,0, ESym("a"), ENum(3)), ESym("b")])),
                Def(0,0, ESym("r"), Apply(0,0, ESym("f"), [ENum(33)])),

                EList([ESym("r"), ESym("a")])
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, VList([33, 3]))

    def test_callcc1(self):
        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([ESym("return")]), [ Apply(0,0, ESym("return"), [ENum(2)]), ENum(4) ])),

                Def(0,0, ESym("r1"), Apply(0,0, ESym("f"), [ELambda(EList([ESym("x")]), ESym("x")) ])),
                Def(0,0, ESym("r2"), Apply(0,0, ESym("call/cc"), [ESym("f")])),

                EList([ESym("r1"), ESym("r2")])
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, VList([4, 2]))

    def test_callcc_yin_yang_raises(self):
        def app(f, n):
            return Apply(0,0, ESym(f), [n])

        class _MyAbort(Exception):
            pass

        def display(args):

            display.res += args.v[0].v

            display.cnt += 1
            if display.cnt > 26:
                raise _MyAbort()

            return ENil()

        display.cnt = 0
        display.res = ""

        self.env.add_builtin(lambda x:x, "display", ["&rest", "args"], display )


        p = Apply(0,0, ESym("begin"), [
                Apply(0,0, ELambda(EList([ESym("yin")]), [
                    Def(0,0, ESym("yin"),
                        Apply(0,0, 
                            ELambda(EList([ESym("cc")]), [app("display", [Str(0,0, "@")]), ESym("cc") ]), [Apply(0,0,ESym("call/cc"), [ELambda(EList([ESym("c")]), ESym("c")) ])])),
                    Apply(0,0, ELambda(EList([ESym("yang")]), [
                        Def(0,0, ESym("yang"),
                            Apply(0,0, 
                                ELambda(EList([ESym("cc")]), [app("display", [Str(0,0, "-")]), ESym("cc") ]), [Apply(0,0,ESym("call/cc"), [ELambda(EList([ESym("c")]), ESym("c")) ])])),
                        Apply(0,0, ESym("yin"), [ESym("yang")])
                    ]),[ENum(0)]),
                ]),[ENum(0)]),
            ])

        with self.assertRaises(_MyAbort):
            eval1(self.env, p)

        self.assertEqual(display.res, "@-@--@---@----@-----@------")

    def test_call_call(self):
        def add(s, n):
            return Apply(0,0, ESym("+"), [ESym(s), ENum(n)])

        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("d"), ELambda(EList([]), ESym("+"))),
                Def(0,0, ESym("r"), Apply(0,0, Apply(0,0, ESym("d"), []), [ENum(33),ENum(9)])),
                ESym("r")
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 42)

    def test_callcc_call(self):
        def add(s, n):
            return Apply(0,0, ESym("+"), [ESym(s), ENum(n)])

        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("cc"), ELambda(EList([ESym("x")]), Apply(0,0,ESym("x"),[ENum(42)]))),
                Def(0,0, ESym("d"), ELambda(EList([]), ESym("cc"))),
                Def(0,0, ESym("r"), Apply(0,0, ESym("call/cc"), [Apply(0,0, ESym("d"), [])])),
                ESym("r")
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 42)

    def test_callcc2(self):
        def add(a, b):
            return Apply(0,0, ESym("+"), [a, b])

        p = ([
#        p = Apply(0,0, ESym("begin"), [

                Def(0,0, ESym("x"), ENum(0)),

                Apply(0,0, ESym("+"), [
                            ENum(1),
                            Apply(0,0, ESym("call/cc"), [
                                    ELambda(EList([ESym("cc")]), [
                                                Def(0,0, ESym("x"), ESym("cc")),
                                                add(ENum(2), Apply(0,0, ESym("cc"), [ENum(3)]))
                                        ])
                            ])
                        ]),

                Apply(0,0,ESym("debug"), [ENum(101)]),

                Apply(0,0, ESym("x"), [ENum(5)]),

                Apply(0,0,ESym("debug"), [ENum(102)]),

                Apply(0,0, ESym("x"), [ENum(6)]),

                Apply(0,0, ESym("debug"), [ENil()]),
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, VList([4, 6, 7]))


if __name__ == "__main__":
    unittest.main()
