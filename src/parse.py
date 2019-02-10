#!/usr/bin/env python3

import yaml
from etypes import *

class ParseError(Exception):
    pass

def comp_to_ast(comp):

    def list_or_one(items):
        if len(items) == 1:
            return items[0]
        return items

    def ast_define(s, e, items):
        l = len(items)
        if l < 2:
            print(s, end='')
            print("-- 'define' requires 2 arguments or more, but has {}".format(l))
            if l < 2:
                print(e, end='')
                print("-- unexpected item")
            raise ParseError()
        sym = items[0]
        if not isinstance(sym, Sym):
            print(sym.start, end='')
            print("-- first 'define' parameter must be a symbol")
            raise ParseError()
        return Def(s, e, sym, items[1:])

    def ast_if(s, e, items):
        l = len(items)
        if l not in [2, 3]:
            print(s, end='')
            print("-- 'if' requires 2 or 3 arguments, but has {}".format(l))
            if l < 2:
                print(e, end='')
                print("-- unexpected item")
            else:
                print(items[3].start, end='')
                print("-- unexpected item")
            raise ParseError()

        els = items[2] if len(items) > 2 else Nil(items[1].start, e)
        return If(s, e, items[0], items[1], els)

    def ast_sym_list(item):
        s = item.start_mark
        e = item.end_mark
        r = []
        for p in map(astize, item.value):
            if not isinstance(p, Sym):
                print(p.start, end='')
                print("-- 'lambda' parameter must be a symbol")
                raise ParseError()
            r.append(p)
        return List(s, e, r)

    def ast_lambda(s, e, items):
        l = len(items)
        if l < 2:
            print(s, end='')
            print("-- 'lambda' requires 2 or more arguments, but has {}".format(l))
            print(e, end='')
            print("-- unexpected item")
            raise ParseError()
        args = ast_sym_list(items[0])
#        exps = list(map(astize, items[1:]))
#        exps = Apply(0,0, Sym(s,s,"begin"), list(map(astize, items[1:])))
#        exps = List(items[1].start_mark, items[-1].end_mark, list(map(astize, items[1:])))
        exps = list_or_one(list(map(astize, items[1:])))
        return Lambda(s, e, args, exps)

    def ast_apply(s, e, fst, items):
        return Apply(s, e, fst, items)

    def ast_call(s, e, items):
        l = len(items)
        if l != 2:
            print(s, end='')
            print("-- 'call' requires 2 arguments: [what [args]], but has {}".format(l))
            print(e, end='')
            print("-- unexpected item")
            raise ParseError()
        what = items[0]
        args = items[1]

        if not isinstance(args, (List, Sym)):
            print(args.start, end='')
            print("-- second argument of 'call' must be a list or a symbol: but got {}".format(args))
            raise ParseError()

        if isinstance(args, List):
            args = args.v
        else:
            args = args
        print(what, args)
        return ast_apply(s, e, what, args)

    def seq2ast(s, e, items):
        if len(items) > 0:
            fst = astize(items[0])
            other = map(astize, items[1:])
            if isinstance(fst, Sym):
                if fst.v == "define":
                    return ast_define(s, e, list(other))
                if fst.v == "if":
                    return ast_if(s, e, list(other))
                if fst.v == "lambda":
                    return ast_lambda(s, e, items[1:])
                if fst.v == "call":
                    return ast_call(s, e, list(other))
                return ast_apply(s, e, Sym.derive(fst, fst.v), list(other))

        return List(s, e, list(map(astize, items)))


    def astize(item):
        s = item.start_mark
        e = item.end_mark

        if isinstance(item, yaml.ScalarNode):
            # double quotes - always a string
            if item.tag == "tag:yaml.org,2002:str" and item.style == '"':
                return Str(s, e, item.value)
            # single quote - string if has double quote
            if item.tag == "tag:yaml.org,2002:str" and item.style == "'" and '"' in item.value:
                return Str(s, e, item.value)

            # single quote - symbol
            if item.tag == "tag:yaml.org,2002:str" and item.style == "'":
                return Sym(s, e, item.value)
            # no quotes - symbol
            if item.tag == "tag:yaml.org,2002:str" and item.style is None:
                return Sym(s, e, item.value)

            if item.tag == "tag:yaml.org,2002:int":
                return Num(s, e, int(item.value))

            if item.tag == "tag:yaml.org,2002:null":
                return Nil(s, e)

            print(s, end='')
            print("-- unexpected yaml scalar")
            raise ParseError()

        if isinstance(item, yaml.SequenceNode):
            return seq2ast(s, e, item.value)

        print(s, end='')
        print("-- unexpected yaml node")
        raise ParseError()

    return astize(comp)


def parse_to_ast(doc):
    c = yaml.compose(doc)
    return comp_to_ast(c)


if __name__ == "__main__":
    doc = """---
- begin
- - define
  - fac'
  - - lambda
    - - x
      - acc
    - - if
      - - '='
        - x
        - 1
      - acc
      - - fac'
        - - '-'
          - x
          - 1
        - - '*'
          - acc
          - x

- - define
  - fac
  - - lambda
    - - x
    - - fac'
      - x
      - 1

- - fac
  - 5000
  - null
"""

    try:
        p = parse_to_ast(doc)
        print(p.format())
    except ParseError:
        print("Aborting")
