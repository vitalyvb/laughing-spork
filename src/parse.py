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

    def seq2ast(s, e, items, top_level=False):
        if len(items) > 0:
            fst = astize(items[0])
            other = map(astize, items[1:])
#            if isinstance(fst, Sym):
#                if fst.v == "lambda":
#                    return ast_lambda(s, e, items[1:])

            if top_level:
                return EvalList(s, e, list(map(astize, items)))
            else:
                return ast_apply(s, e, fst, list(other))

        return List(s, e, [])


    def astize(item, top_level=False):
        s = item.start_mark
        e = item.end_mark

        if isinstance(item, yaml.ScalarNode):
#            console.log(item.tag)
#            console.log(item.style)
#            console.log(item.value)
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
            return seq2ast(s, e, item.value, top_level)

        print(s, end='')
        print("-- unexpected yaml node")
        raise ParseError()

    return astize(comp, True)


def parse_to_ast(doc, init=None):
    if init is None:
        init = []
    else:
        init = init.v
    c = yaml.compose(doc)
    ast = comp_to_ast(c)
    if isinstance(ast, EvalList):
        init.extend(ast.v)
        res = Module(0,0, init)
    else:
        init.append(ast)
        res = Module(0,0, init)
    return res 


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
