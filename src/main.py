#!/usr/bin/env python3

import sys
from parse import parse_to_ast, ParseError
from eval import get_prelude_env, VMEval

def main():
    if len(sys.argv) < 2:
        raise Exception("usage: main.py file.yml")

    doc = open(sys.argv[1], "r").read()
    try:
        prog = parse_to_ast(doc)
        print("-----------")
        print(prog.format())
        print("-----------")
    except ParseError:
        print("Parse error.")
        sys.exit(1)

#    print(prog.args[0].exp.exp[0].format())

    env = get_prelude_env()
    eval1 = VMEval()
    try:
        x = eval1(env, prog)
        print("Result = {}".format(x))
    finally:
        eval1.stats.print()

if __name__ == '__main__':
#    import cProfile
#    pr = cProfile.Profile()
#    pr.enable()
    main()
#    pr.disable()
#    pr.print_stats()
