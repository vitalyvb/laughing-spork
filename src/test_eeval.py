#!/usr/bin/env python3

from etypes import *
from eeval import *

import unittest
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
