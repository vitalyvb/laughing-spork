#!/usr/bin/env python3

from functools import reduce
import time

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

def eval1(env, exp):


    while True:

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

#            print(exp.start)
            time.sleep(0.3)

        if isinstance(exp, Sym):
            exp = env[exp.v]

        if isinstance(exp, If):
            v = eval1(env, exp.exp)
            if not isinstance(v, Nil):
                exp = exp.thn
            else:
                exp = exp.els
            continue

        if isinstance(exp, Lambda):
            return Closure(env.copy(), exp.params, exp.exp)

        if isinstance(exp, (Nil, Num, Str, Lambda, Closure)):
            return exp

        if isinstance(exp, List):
            r = []
            env2 = env.copy()
            for i in exp.v:
                r.append(eval1(env2, i))
            return EList(r)

        if isinstance(exp, list):
            tail = exp[-1]
            env2 = env.copy()
            for i in exp[:-1]:
                eval1(env2, i)
            env = env2
            exp = tail

            continue

        if isinstance(exp, Def):
            env[exp.sym.v] = eval1(env, exp.exp)
            return ENil()

        if callable(exp):
            return exp(eval1, env)

        if isinstance(exp, Apply):
            f = env[exp.sym.v]
            env2 = env.copy()

            try:
                for n, x in enumerate(f.params.v):
                    if x.v == "&rest":
                        p = n
                        break
                else:
                    raise ValueError()
                args = f.params.v[:p]
                argv = f.params.v[p+1]

            except ValueError:
                args = f.params.v
                argv = None

            if len(args) > len(exp.args):
                raise Exception("not enough arguments to call {}".format(f))


            for arg, val in zip(args, exp.args):
                env2[arg.v] = eval1(env, val)


            if argv is not None:
                agg = []
                for val in exp.args[len(args):]:
                    agg.append(val)

                env2[argv.v] = EList(agg)

            if isinstance(f, Closure):
                # why the hell this works at all?
                # because of no variable names conflict?
                env2.update(f.env)

            env = env2
            exp = f.exp
            continue


        print(exp)
        raise Exception("Eval '{}' not implemented".format(type(exp)))


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

        def value(evl, env):
#            print(env)
            fargs = list(map(lambda v: evl(env, ESym(v)), args))
            return tr(func(*tuple(fargs)))

        prelude[s] = ELambda(EList(list(map(ESym, ar))), value)

    _id = lambda x:x

    define(_id, "begin", ["&rest", "args"], lambda args: reduce(lambda _,b: b, args.v) )

#    define(_id, "debug", ["&rest", "args"], lambda args: (ENil(), print(args.list))[0] )

#    define(ENum, "+", ["&rest", "args"], lambda args: reduce(lambda a,b: a+b.value, args.list, 0) )
    define(ENum, "*", ["&rest", "args"], lambda args: reduce(lambda a,b: a*b.v, args.v, 1) )
    define(ENum, "-", ["i", "&rest", "args"], lambda i, args: reduce(lambda a,b: a-b.v, args.v, i.v) )
#    define(ENum, "/", ["i", "&rest", "args"], lambda i, args: reduce(lambda a,b: a//b.value, args.list, i.value) )

    define(_id, "=", ["a", "b"], lambda a, b: [ENil(), EList([])][a.v == b.v] )

    return prelude

