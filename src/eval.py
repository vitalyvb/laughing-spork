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
    return List(0, 0, v)

def ELambda(v, e):
    return Lambda(0, 0, v, e)


class Frame(object):
    def __init__(self, env, data, cont):
        self.env = env
        self.data = data
        self.cont = cont

recurs = 0

def eval1(env, exp):
    global recurs
    recurs += 1

#    print("eval1 recursion level: {}".format(recurs))

    try:
        return eval2(env, exp)
    finally:
        recurs -= 1


def eval2(env, exp):

    stack = deque()

    def s_push(cont, data):
        stack.append(Frame(env, data, cont))

    def s_pop():
        frame = stack.pop()
        return frame


#    def op_if(env, exp):
#        v = eval1(env, exp.exp)
#        if not isinstance(v, Nil):
#            exp = exp.thn
#        else:
#            exp = exp.els
#        return exp

    def op_if(env, exp):
        s_push(op_if2, exp)
        return exp.exp

    def op_if2(env, exp, _cexp, cdata):
        if not isinstance(exp, Nil):
            exp = cdata.thn
        else:
            exp = cdata.els
        return exp


#    def op_def(env, exp):
#        env[exp.sym.v] = eval1(env, exp.exp)
#        return ENil()

    def op_def(env, exp):
        s_push(op_def2, exp)
        return exp.exp

    def op_def2(env, exp, _cexp, cdata):
        env[cdata.sym.v] = exp
        return ENil()


#    def op_list(env, exp):
#        r = []
#        env2 = env.copy()
#        for i in exp.v:
#            r.append(eval1(env2, i))
#        return EList(r)


    def op_list(env, exp):
        if len(exp.v) == 0:
            return (exp,)
        s_push(op_list2, (exp, []))
        return exp.v[0]

    def op_list2(env, exp, _cexp, cdata):
        cdata, acc = cdata
        acc.append(exp)

        if len(acc) >= len(cdata.v):
            return (EList(acc),)

        s_push(op_list2, (cdata, acc))
        return cdata.v[len(acc)]


#    def op_pylist(env, exp):
#        tail = exp[-1]
#        for i in exp[:-1]:
#            eval1(env, i)
#        exp = tail
#        return tail

    def op_pylist(env, exp):
        if len(exp) > 1:
            s_push(op_pylist2, (exp, []))
        return exp[0]

    def op_pylist2(env, exp, _cexp, cdata):
        cdata, acc = cdata
        acc.append(exp)

        if len(acc) >= len(cdata)-1:
            return cdata[len(acc)]

        s_push(op_pylist2, (cdata, acc))
        return cdata[len(acc)]

#    def op_apply(env, exp):
#        f = env[exp.sym.v]
#
#        try:
#            for n, x in enumerate(f.params.v):
#                if x.v == "&rest":
#                    p = n
#                    break
#            else:
#                raise ValueError()
#            args = f.params.v[:p]
#            argv = f.params.v[p+1]
#
#        except ValueError:
#            args = f.params.v
#            argv = None
#
#        if len(args) > len(exp.args):
#            raise Exception("not enough arguments to call {}".format(f))
#
#        for arg, val in zip(args, exp.args):
#            env[arg.v] = eval1(env, val)
#
#        if argv is not None:
#            agg = []
#            for val in exp.args[len(args):]:
#                # not evaluated, probably bug, but needed for tail call optimization for begin
#                agg.append(val)
#
#            env[argv.v] = EList(agg)
#
#        if isinstance(f, Closure):
#            # why the hell this works at all?
#            # because of no variable names conflict?
#            env.update(f.env)
#
#        return f.exp

    def op_apply(env, exp):
        if exp.sym.v == "begin":
            return exp.args

        f = env[exp.sym.v]

        for n, x in enumerate(f.params.v):
            if x.v == "&rest":
                p = n
                args = f.params.v[:p]
                argv = f.params.v[p+1]
                break
        else:
            args = f.params.v
            argv = None


        if len(args) > len(exp.args):
            raise Exception("not enough arguments to call {}".format(f))

        ctx = (f, args, argv)
        largs = exp.args

        s_push(op_apply2, ctx)
        return EList(largs)

    def op_apply2(env, exp, _cexp, cdata):
        (f, args, argv) = cdata

        for arg, val in zip(args, exp.v):
            env[arg.v] = val

        if argv is not None:
            env[argv.v] = EList(exp.v[len(args):])

        if isinstance(f, Closure):
            # why the hell this works at all?
            # because of no variable names conflict?
            env.update(f.env)

        return f.exp

    def op_callable(env, exp):
        return exp(env)


    while True:

        while True:

#            if len(stack) > 1:
#                print("my stack size: {}".format(len(stack)))

            if 0 and hasattr(exp, 'start') and exp.start != 0:
                l = exp.start.buffer.splitlines()
                ln = exp.start.line
                cn = exp.start.column

                if isinstance(exp, Sym) and exp.v in env and isinstance(env[exp.v], (Num,Str)):
                    val = "--  {} = {}".format(exp.v, env[exp.v].v)
                else:
                    val = ""

                print("{:3}: {}".format(ln-2, l[ln-2]))
                print("{:3}: {}".format(ln-1, l[ln-1]))
                print("{:3}: {}".format(ln, l[ln]))
                print("     "+" "*cn+"^"+val)

#                print(exp.start)
                time.sleep(0.3)


            if isinstance(exp, Sym):
                exp = env[exp.v]

            if isinstance(exp, (Nil, Num, Str, Closure)):
                break

            if isinstance(exp, Apply):
                exp = op_apply(env, exp)
                continue

            if isinstance(exp, List):
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

            if isinstance(exp, If):
                exp = op_if(env, exp)
                continue

            if isinstance(exp, Lambda):
                exp = Closure(env.copy(), exp.params, exp.exp)
                break

            if isinstance(exp, Def):
                exp = op_def(env, exp)
                continue

            print(exp)
            raise Exception("Eval '{}' not implemented".format(type(exp)))

        if len(stack) == 0:
            break

        frame = s_pop()
        env = frame.env
        exp = frame.cont(env, exp, None, frame.data)

    return exp


def get_prelude_env():
    prelude = {
        "&rest": type("Rest", (object,), {}),
    }

    def define(tr, s, ar, func):
        args = []
        argv = None
        for i, v in enumerate(ar):
            if v == "&rest":
                break
            args.append(v)

        if v == "&rest":
            argv = ar[i+1]

#        print(s)
#        print(args)
#        print(argv)
#        print(".")

        if argv is not None:
            args.append(argv)

        def value(env):
            fargs = list(map(lambda v: env[v], args))
            return tr(func(*tuple(fargs)))

        prelude[s] = ELambda(EList(list(map(ESym, ar))), value)

    _id = lambda x:x

#    prelude["begin"] = ELambda(EList(list(map(ESym, ["&rest", "args"]))), lambda evl, env: env["args"].v)

#    define(_id, "debug", ["&rest", "args"], lambda args: (ENil(), print(args.list))[0] )

    define(ENum, "+", ["&rest", "args"], lambda args: reduce(lambda a,b: a+b.v, args.v, 0) )
    define(ENum, "*", ["&rest", "args"], lambda args: reduce(lambda a,b: a*b.v, args.v, 1) )
    define(ENum, "-", ["i", "&rest", "args"], lambda i, args: reduce(lambda a,b: a-b.v, args.v, i.v) )
    define(ENum, "/", ["i", "&rest", "args"], lambda i, args: reduce(lambda a,b: a//b.v, args.v, i.v) )

    define(_id, "=", ["a", "b"], lambda a, b: [ENil(), EList([])][a.v == b.v] )

    return prelude


class Test_Eval(unittest.TestCase):

    def setUp(self):
        self.env = get_prelude_env()


    def test_unhandled1(self):
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


    def test_sub1(self):
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


    def test_div1(self):
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

    @unittest.skip("not implemented")
    def test_div5(self):
        p = Apply(0,0, ESym("/"), [ENum(1), ENum(0)])
        with self.assertRaises(EvalRuntimeError):
            eval1(self.env, p)


    def test_eq1(self):
        p = Apply(0,0, ESym("="), [])
        with self.assertRaises(Exception):
            eval1(self.env, p)

    def test_eq2(self):
        p = Apply(0,0, ESym("="), [ENum(44)])
        with self.assertRaises(Exception):
            eval1(self.env, p)

    def test_eq3(self):
        p = Apply(0,0, ESym("="), [ENum(3), ENum(5)])
        res = eval1(self.env, p)
        self.assertIsInstance(res, Nil)

    def test_eq4(self):
        p = Apply(0,0, ESym("="), [ENum(3), ENum(3)])
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

    def test_lambda3(self):
        p = Apply(0,0, ESym("begin"), [
                Def(0,0, ESym("f"), ELambda(EList([ESym("x")]), ESym("x"))),
                Apply(0,0, ESym("f"), [])
            ])

        with self.assertRaises(Exception):
            eval1(self.env, p)

    @unittest.skip("bug")
    def test_lambda4(self):
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

    def test_lambda7(self):
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
                    If(0,0, Apply(0,0, ESym("="), [ESym("x"), ENum(0)]),
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
                        If(0,0, Apply(0,0, ESym("="), [ESym("x"), ENum(0)]),
                                ESym("acc"),
                                Apply(0,0, ESym("f"), [add("acc", ESym("x")),
                                                   add("x", ENum(-1))]))
                    ]))),
                Apply(0,0, ESym("f"), [ENum(0), ENum(1000)])
            ])

        res = eval1(self.env, p)
        self.assertEqual(res, 500500)


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

    @unittest.skip("bug")
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


if __name__ == "__main__":
    unittest.main()
