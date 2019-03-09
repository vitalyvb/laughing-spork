#!/usr/bin/env python3

import sys
from parse import parse_to_ast, ParseError
from eeval import get_prelude_env, VMEval
import re

def log(s):
    try:
        console.log(s)
    except:
        print(s)

init_doc = """---

- - defmacro
  - lambda

  - - '&lambda'
    - - quote
      - args
      - '&rest'
      - code

    - - '&apply'
      - '&lambda'
      - - quote
        - - '&apply.tolist'
          - args
        - - '&list.eval'
          - code


- - defmacro
  - if
  - - lambda
    - - expr
      - then
      - else

    - - '&apply'
      - '&if'
      - - list
        - expr
        - - '&lambda'
          - []
          - then

        - - '&lambda'
          - []
          - else


- - defmacro
  - defun
  - - lambda
    - - name
      - args
      - '&rest'
      - code

    - - eval
      - - '&apply'
        - define
        - - list
          - name
          - - '&lambda'
            - - '&apply.tolist'
              - args
            - - '&list.eval'
              - code


- - defmacro
  - cond

  - - lambda
    - - expr
      - then
      - '&rest'
      - cont

    - - '&if'
      - - empty?
        - cont
      - - lambda
        - []

        - - callmacro
          - 'if'
          - - list
            - expr
            - then
            - null

      - - lambda
        - []

        - - callmacro
          - 'if'
          - - list
            - expr
            - then
            - - callmacro
              - cond
              - cont


"""

default_doc = """---

- - defun
  - fac'
  - - x
    - acc
  - - if
    - - eq?
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

- - defun
  - fac
  - - x
  - - fac'
    - x
    - 1

- - fac
  - 10

"""

def main(doc):

#    try:
#        init = parse_to_ast(init_doc)
#        print(init)
#    except Exception as e:
#        console.log(" ***** EXCEPTION *****")
#        console.log(e)
#    return

    try:
        init = parse_to_ast(init_doc)
        prog = parse_to_ast(doc, init)
#        print("-----------")
#        print(prog.format())
#        print("-----------")
    except ParseError:
        return "Parse error: " + str(e.__args__)

    env = get_prelude_env()
    eval1 = VMEval()
    try:
        x = eval1.__call__(env, prog)
        print("Result = {}".format(x))
        return str(x)
    except Exception as e:
        log(e)
        return str(e.__args__)
    finally:
        eval1.stats.print()

#if __name__ == '__main__':
#    main(open(sys.argv[1], "r").read())
