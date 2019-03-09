#!/usr/bin/env python3

#from functools import reduce
import time
#from collections import deque

from etypes import *

def reduce(func, iterable, res=None):
    iterator = iter(iterable)
    if res is None:
        res = next(iterator)
    for arg in iterator:
        res = func(res, arg)
    return res

def log(s):
    try:
        console.log(s)
    except:
        print(s)

class deque(object):
    def __init__(self, a=None, maxlen=0):
        if a is None:
            a = []
        elif isinstance(a, deque):
            a = a.lst
        self.lst = list(a)
        self._maxlen = maxlen

        for x in ["remove","__repr__","__str__","__reduce_ex__","__reduce__","__getitem__"]:
            setattr(self,x,getattr(self.lst,x))

    def __len__(self):
        return len(self.lst)

    def __iter__(self):
        return self.lst.__iter__()

    def pop(self):
        return self.lst.pop()

    def popleft(self):
        x = self.lst[0]
        del self.lst[0]
        return x

    def extend(self, other):
        if isinstance(other, deque):
            other = other.lst
        for i in other:
            self.append(i)

    def append(self, item):
        if self._maxlen > 0:
            if len(self.lst) >= self._maxlen:
                del self.lst[0]
        self.lst.append(item)

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

    def format(self, align):
        return align(self.__class__.__name__)

class _Rest(EvalQuoted):
    pass
class _Begin(EvalQuoted):
    pass
class _Call(EvalQuoted):
    pass
class _CallCC(EvalQuoted):
    pass
class _Eval(EvalQuoted):
    pass
class _Quote(EvalQuoted):
    pass
class _Define(EvalQuoted):
    pass
class _EIf(EvalQuoted):
    pass
class _DefMacro(EvalQuoted):
    pass
class _CallMacro(EvalQuoted):
    pass

Rest =      _Rest()
Begin =     _Begin()
Call =      _Call()
CallCC =    _CallCC()
Eval =      _Eval()
Quote =     _Quote()
Define =    _Define()
EIf =       _EIf()
DefMacro =  _DefMacro()
CallMacro = _CallMacro()

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

                if isinstance(exp, Sym) and env.isin(exp.v) and isinstance(env.get(exp.v), (Num,Str)):
                    val = "--  {} = {}".format(exp.v, env.get(exp.v).v)
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
            log("    {}:".format(pos))
            txt = self.traces.popleft()
            log("\n".join("        "+l for l in txt))
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

def eval1(env, exp):
    e = VMEval(trace=10)
    return e.__call__(env,exp)

def eval2(vm, stack, env, exp1):

    def s_push(env, cont, data, descr=None, src=None):
        stack.append(Frame(env, data, cont, descr, src))

    def s_reset(st):
        while len(stack)>0:
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
        acc = list(acc)
        acc.append(exp)
#        acc = acc + [exp]
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
        acc = list(acc)
        acc.append(exp)

        if len(acc) >= len(cdata)-1:
#            env.weaken()
            return cdata[len(acc)]

        s_push(env, op_pylist2, (cdata, acc), descr="pylist:N", src=exp)
        return cdata[len(acc)]

    def callcc(envn):
        res = envn.get("&cont")
        (stack0, env0) = envn.get("&closure")

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
        return Apply(0,0, env.get("!"+cdata.v).exp, list(Apply(0,0, Quote, [x]) for x in f.v))

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
            raise Exception("too many arguments to call {}, have {}, need {}".format(f, len(exp.v), len(args)))

        alls = list(args)
        if argv is not None:
            alls.append(argv)

        if isinstance(f, Closure):
            env = f.env[1]

        env = Env(env, [x.v for x in alls])
        frame.env = env

        for arg, val in zip(args, exp.v):
            env.set(arg.v, val)

        if argv is not None:
            env.set(argv.v, VList(exp.v[len(args):]))

        if isinstance(f, Closure):
            env.set("&closure", (f.env[0], env))

        return f.exp

    def op_callable(env, exp):
        return exp(env)



    exp = exp1
    while True:


        while True:

#            print(exp)

            if len(stack) > 40:
                print("my stack size: {}".format(len(stack)))
                for x in stack:
                    print("{:12s} - {}".format(x.descr, x))
                1/0
            vm.stats.stack(len(stack))
            vm.stats.envd(env)

            vm.add_trace_point(exp, env)


            if isinstance(exp, Sym):
                #log(exp.v)
                if env.isin("!"+exp.v):
                    exp = env.get("!"+exp.v)
                else:
                    exp = env.get(exp.v)
                #log(exp)
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
        e._locals = {}
        for k in list(self._locals.keys()):
            e._locals[k] = self._locals[k]
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

    def search(self, s, default):
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
        d = self.search(s, None)
        if d is not None:
            return d[s]
        else:
            raise Exception("symbol '{}' not defined".format(s))

    def isin(self, i):
        d = self.search(i, None)
        return d is not None and i in d

#    def __contains__(self, i):
#        return self.isin(i)

#    def __getitem__(self, i):
#        return self.get(i)

#    def __setitem__(self, i, v):
#        return self.set(i, v)

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
                fargs = list(map(lambda v: env.get(v), args))
                return (tr(func(*tuple(fargs))),)
            val = ELambda(VList(list(map(ESym, ar))), value)
        elif rawfunc is not None:
            val = rawfunc
#        val.format = lambda align:align(s)
#        setattr(val,"format",lambda align:align(s))
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

    define(_id, "eq?", ["a", "b"], lambda a, b: [ENil(), VList([])][1 if a.v == b.v else 0] )

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

    define(_id, "list?", ["x"], lambda x: [ENil(), VList([])][1 if isinstance(x, List) else 0] )

    define(ENum, "length", ["lst"], lambda lst: len(lst.v) )
    define(_id, "empty?", ["lst"], lambda lst: [ENil(), VList([])][1 if isinstance(lst, List) and len(lst.v) == 0 else 0] )

    define(_id, "elem", ["lst", "idx"], lambda lst, idx: lst.v[idx.v] )

    define(_id, "&apply", ["exp", "args"], lambda exp, args: Apply(0,0, exp, args.v) )
    define(_id, "&apply?", ["app"], lambda app: [ENil(), VList([])][q if isinstance(app, Apply) else 0] )
    define(_id, "&apply.exp", ["app"], lambda app: app.sym )
    define(_id, "&apply.args", ["app"], lambda app: VList(app.args) )
    def _app_tolist(app):
        # for lambda - empty list in not apply
        if isinstance(app, Apply):
            l = [app.sym]
            l.extend(app.args)
            return VList(l)
        return app
    define(_id, "&apply.tolist", ["app"], _app_tolist )

    define(_id, "&list.elist", ["lst"], lambda lst: EList(lst.v) )
    define(_id, "&list.eval", ["lst"], lambda lst: lst.v )

    define(_id, "&lambda", ["args", "exp"], lambda args, exp: Lambda(0,0, args, exp) )

    define(_id, "&format", ["exp"], lambda exp: (ENil(), print(exp.format()))[0] )

    def eif(env):
        if not isinstance(env.get("exp"), Nil):
            exp = env.get("fthen")
        else:
            felse = env.get("felse").v
            if len(felse) == 1:
                exp = felse[0]
            elif len(felse) > 1:
                raise EvalRuntimeError("too may elses in if")
            else:
                return ENil()
#        env.weaken()
        return exp.exp
#        return Apply(0,0, exp, [])

#    define(_id, "&if", ["exp", "fthen", "&rest", "felse"], rawfunc=ELambda(VList(list(map(ESym, ["exp", "fthen", "&rest", "felse"]))),eif) )

    return new_env


def _lmb_If(x):
    return Lambda(0,0, VList([]), x)

def If(_s, _e, x, t, e=None):
    if e is None:
        return Apply(0,0, ESym("&if"), [x, _lmb_If(t)])
    return Apply(0,0, ESym("&if"), [x, _lmb_If(t), _lmb_If(e)])

