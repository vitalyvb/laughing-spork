#!/usr/bin/env python3

from collections import namedtuple

class Token(object):
    def __init__(self, start, end):
        self._start = start
        self._end = end

    @classmethod
    def derive(cls, tok):
        return cls(tok.start, tok.end)

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    def __repr__(self):
        return "<{}>".format(self.__class__.__name__)

    def format(self, align=lambda x:x):
        return align(repr(self))

    def __hash__(self):
        raise Exception()

    def __eq__(self, other):
        raise Exception("can't compare {} with {}".format(self.__class__.__name__, other))


class Nil(Token):
    pass


class _Value(Token):
    def __init__(self, start, end, v):
        super(_Value, self).__init__(start, end)
        self._v = v

    @classmethod
    def derive(cls, tok, v):
        return cls(tok.start, tok.end, v)

    @property
    def v(self):
        return self._v

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, repr(self._v))

#    def __hash__(self):
#        return self._v.__hash__()

    def __eq__(self, other):
        if isinstance(other, _Value):
            return self._v.__eq__(other._v)
        return self._v.__eq__(other)

class Num(_Value):
    pass


class Sym(_Value):
    pass


class Str(_Value):
    pass

def relist(exp):
    if not isinstance(exp, list):
        return exp

    if len(exp) > 1:
        s = exp[0].start
        e = exp[-1].end
    else:
        s = 0
        e = 0

    return List(s, e, exp, style=("[", "]"))

class List(_Value):
    def __init__(self, *a, style=("","."), **kw):
        super(List, self).__init__(*a, **kw)
        self._style = style

    def format(self, align=lambda x:x):
        if self._v == []:
            return align("[]")

        def align2(x):
            return "    " + align(x)

        r = []
        r.append(align2(self._style[0]))
        r.extend(x.format(align2) for x in self._v)
        r.append(align2(self._style[1]))

        return "\n".join(r)


class Def(Token):
    def __init__(self, start, end, sym, exp):
        super(Def, self).__init__(start, end)
        self._sym = sym
        self._exp = exp

    @classmethod
    def derive(cls, tok, sym, exp):
        return cls(tok.start, tok.end, sym, exp)

    @property
    def sym(self):
        return self._sym

    @property
    def exp(self):
        return self._exp

    def format(self, align=lambda x:x):

        def align2(x):
            return "    " + align(x)

        r = []
        r.append(self._sym.format(align))
        r.append(align(":="))
        r.append(relist(self._exp).format(align2))

        return "\n".join(r)


class If(Token):
    def __init__(self, start, end, exp, thn, els):
        super(If, self).__init__(start, end)
        self._exp = exp
        self._thn = thn
        self._els = els

    @classmethod
    def derive(cls, tok, sym, exp):
        return cls(tok.start, tok.end, sym, exp)

    @property
    def exp(self):
        return self._exp

    @property
    def thn(self):
        return self._thn

    @property
    def els(self):
        return self._els

    def format(self, align=lambda x:x):

        def align2(x):
            return "    " + align(x)

        r = []
        r.append(align("if"))
        r.append(self._exp.format(align2))
        r.append(align("then"))
        r.append(self._thn.format(align2))
        r.append(align("else"))
        r.append(self._els.format(align2))

        return "\n".join(r)


class Apply(Token):

    def __init__(self, start, end, sym, args):
        super(Apply, self).__init__(start, end)
        self._sym = sym
        self._args = args

    @property
    def sym(self):
        return self._sym

    @property
    def args(self):
        return self._args

    def format(self, align=lambda x:x):

        def align2(x):
            return "  " + align(x)

        r = []
        r.append(self._sym.format(align)+"(")
        r.extend(x.format(align2) for x in self._args)
        r.append(align(")"))

        return "\n".join(r)


class Lambda(Token):

    def __init__(self, start, end, params, exp):
        super(Lambda, self).__init__(start, end)
        self._params = params
        self._exp = exp

    @classmethod
    def derive(cls, tok, *args, **kw):
        return cls(tok.start, tok.end, *args, **kw)

    @property
    def exp(self):
        return self._exp

    @property
    def params(self):
        return self._params

    def format(self, align=lambda x:x):

        def align2(x):
            return "    " + align(x)

        r = []
        r.append(align("λ ") + repr(self._params.v))
        r.append(align(":="))
        r.append(relist(self._exp).format(align2))
        r.append(align("."))

        return "\n".join(r)


Closure = namedtuple("Closure", ["env", "params", "exp"])


