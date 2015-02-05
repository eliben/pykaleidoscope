# Chapter 1 & 2 - Lexer and Parser

from collections import namedtuple
from enum import Enum


# Each token is a tuple of kind and value. kind is one of the enumeration values
# in TokenKind. value is the textual value of the token in the input.
class TokenKind(Enum):
    EOF = -1
    DEF = -2
    EXTERN = -3
    IDENTIFIER = -4
    NUMBER = -5
    OPERATOR = -6


Token = namedtuple('Token', 'kind value')


class Lexer(object):
    """Lexer for Kaleidoscope.

    Initialize the lexer with a string buffer. tokens() returns a generator that
    can be queried for tokens. The generator will emit an EOF token before
    stopping.
    """
    def __init__(self, buf):
        assert len(buf) >= 1
        self.buf = buf
        self.pos = 0
        self.lastchar = self.buf[0]

    def tokens(self):
        while self.lastchar:
            # Skip whitespace
            while self.lastchar.isspace():
                self._advance()
            # Identifier or keyword
            if self.lastchar.isalpha():
                id_str = ''
                while self.lastchar.isalnum():
                    id_str += self.lastchar
                    self._advance()
                if id_str == 'def':
                    yield Token(kind=TokenKind.DEF, value=id_str)
                elif id_str == 'extern':
                    yield Token(kind=TokenKind.EXTERN, value=id_str)
                else:
                    yield Token(kind=TokenKind.IDENTIFIER, value=id_str)
            # Number
            elif self.lastchar.isdigit() or self.lastchar == '.':
                num_str = ''
                while self.lastchar.isdigit() or self.lastchar == '.':
                    num_str += self.lastchar
                    self._advance()
                yield Token(kind=TokenKind.NUMBER, value=num_str)
            # Comment
            elif self.lastchar == '#':
                self._advance()
                while self.lastchar and self.lastchar not in '\r\n':
                    self._advance()
            elif self.lastchar:
                # Some other char
                yield Token(kind=TokenKind.OPERATOR, value=self.lastchar)
                self._advance()
        yield Token(kind=TokenKind.EOF, value='')

    def _advance(self):
        try:
            self.pos += 1
            self.lastchar = self.buf[self.pos]
        except IndexError:
            self.lastchar = ''


# AST hierarchy
class ASTNode(object):
    def dump(self, indent=0):
        raise NotImplementedError


class ExprAST(ASTNode):
    pass


class NumberExprAST(ExprAST):
    def __init__(self, val):
        self.val = val

    def dump(self, indent=0):
        return '{0}{1}[{2}]'.format(
            ' ' * indent, self.__class__.__name__, self.val)


class VariableExprAST(ExprAST):
    def __init__(self, name):
        self.name = name

    def dump(self, indent=0):
        return '{0}{1}[{2}]'.format(
            ' ' * indent, self.__class__.__name__, self.name)


class BinaryExprAST(ExprAST):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def dump(self, indent=0):
        s = '{0}{1}[{2}]\n'.format(
            ' ' * indent, self.__class__.__name__, self.op)
        s += self.lhs.dump(indent + 2) + '\n'
        s += self.rhs.dump(indent + 2)
        return s


class CallExprAST(ExprAST):
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args

    def dump(self, indent=0):
        s = '{0}{1}[{2}]\n'.format(
            ' ' * indent, self.__class__.__name__, self.callee)
        for arg in self.args:
            s += arg.dump(indent + 2) + '\n'
        return s[:-1]  # snip out trailing '\n'


class PrototypeAST(ASTNode):
    def __init__(self, name, argnames):
        self.name = name
        self.argnames = argnames

    def dump(self, indent=0):
        return '{0}{1}[{2}]'.format(
            ' ' * indent, self.__class__.__name__, ', '.join(self.argnames))


class FunctionAST(ASTNode):
    def __init__(self, proto, body):
        self.proto = proto
        self.body = body

    def dump(self, indent=0):
        s = '{0}{1}[{2}]\n'.format(
            ' ' * indent, self.__class__.__name__, self.proto.dump())
        s += self.body.dump(indent + 2) + '\n'
        return s


class ParseError(Exception): pass


class Parser(object):
    """Parser for the Kaleidoscope language.

    After the parser is created, invoke parse_toplevel multiple times to parse
    Kaleidoscope source into an AST.
    """
    def __init__(self):
        self.token_generator = None
        self.cur_tok = None

    # toplevel ::= definition | external | expression | ';'
    def parse_toplevel(self, buf):
        """Given a string, returns an AST node representing it."""
        self.token_generator = Lexer(buf).tokens()
        self.cur_tok = None
        self._get_next_token()

        if self.cur_tok.kind == TokenKind.EXTERN:
            return self._parse_external()
        elif self.cur_tok.kind == TokenKind.DEF:
            return self._parse_definition()
        elif self._cur_tok_is_operator(';'):
            self._get_next_token()
            return None
        else:
            return self._parse_toplevel_expression()

    def _get_next_token(self):
        self.cur_tok = next(self.token_generator)

    def _match(self, expected_kind, expected_value=None):
        """Consume the current token; verify that it's of the expected kind.

        If expected_kind == TokenKind.OPERATOR, verify the operator's value.
        """
        if (expected_kind == TokenKind.OPERATOR and
            not self._cur_tok_is_operator(expected_value)):
            raise ParseError('Expected "{0}"'.format(expected_value))
        elif expected_kind != self.cur_tok.kind:
            raise ParseError('Expected "{0}"'.format(expected_kind))
        self._get_next_token()

    _precedence_map = {'<': 10, '+': 20, '-': 20, '*': 40}

    def _cur_tok_precedence(self):
        """Get the operator precedence of the current token."""
        try:
            return Parser._precedence_map[self.cur_tok.value]
        except KeyError:
            return -1

    def _cur_tok_is_operator(self, op):
        """Query whether the current token is the operator op"""
        return (self.cur_tok.kind == TokenKind.OPERATOR and
                self.cur_tok.value == op)

    # identifierexpr
    #   ::= identifier
    #   ::= identifier '(' expression* ')'
    def _parse_identifier_expr(self):
        id_name = self.cur_tok.value
        self._get_next_token()
        # If followed by a '(' it's a call; otherwise, a simple variable ref.
        if not self._cur_tok_is_operator('('):
            return VariableExprAST(id_name)

        self._get_next_token()
        args = []
        if not self._cur_tok_is_operator(')'):
            while True:
                args.append(self._parse_expression())
                if self._cur_tok_is_operator(')'):
                    break
                self._match(TokenKind.OPERATOR, ',')

        self._get_next_token()  # consume the ')'
        return CallExprAST(id_name, args)

    # numberexpr ::= number
    def _parse_number_expr(self):
        result = NumberExprAST(self.cur_tok.value)
        self._get_next_token()  # consume the number
        return result

    # parenexpr ::= '(' expression ')'
    def _parse_paren_expr(self):
        self._get_next_token()  # consume the '('
        expr = self._parse_expression()
        self._match(TokenKind.OPERATOR, ')')
        return expr

    # primary
    #   ::= identifierexpr
    #   ::= numberexpr
    #   ::= parenexpr
    def _parse_primary(self):
        if self.cur_tok.kind == TokenKind.IDENTIFIER:
            return self._parse_identifier_expr()
        elif self.cur_tok.kind == TokenKind.NUMBER:
            return self._parse_number_expr()
        elif self._cur_tok_is_operator('('):
            return self._parse_paren_expr()
        else:
            raise ParseError('Unknown token when expecting an expression')

    # binoprhs ::= (<binop> primary)*
    def _parse_binop_rhs(self, expr_prec, lhs):
        """Parse the right-hand-side of a binary expression.

        expr_prec: minimal precedence to keep going (precedence climbing).
        lhs: AST of the left-hand-side.
        """
        while True:
            cur_prec = self._cur_tok_precedence()
            # If this is a binary operator with precedence lower than the
            # currently parsed sub-expression, bail out. If it binds at least
            # as tightly, keep going.
            # Note that the precedence of non-operators is defined to be -1,
            # so this condition handles cases when the expression ended.
            if cur_prec < expr_prec:
                return lhs
            op = self.cur_tok.value
            self._get_next_token()  # consume the operator
            rhs = self._parse_primary()

            next_prec = self._cur_tok_precedence()
            # There are three options:
            # 1. next_prec > cur_prec: we need to make a recursive call
            # 2. next_prec == cur_prec: no need for a recursive call, the next
            #    iteration of this loop will handle it.
            # 3. next_prec < cur_prec: no need for a recursive call, combine
            #    lhs and the next iteration will immediately bail out.
            if cur_prec < next_prec:
                rhs = self._parse_binop_rhs(cur_prec + 1, rhs)

            # Merge lhs/rhs
            lhs = BinaryExprAST(op, lhs, rhs)

    # expression ::= primary binoprhs
    def _parse_expression(self):
        lhs = self._parse_primary()
        # Start with precedence 0 because we want to bind any operator to the
        # expression at this point.
        return self._parse_binop_rhs(0, lhs)

    # prototype ::= id '(' id* ')'
    def _parse_prototype(self):
        name = self.cur_tok.value
        self._match(TokenKind.IDENTIFIER)
        self._match(TokenKind.OPERATOR, '(')
        argnames = []
        while self.cur_tok.kind == TokenKind.IDENTIFIER:
            argnames.append(self.cur_tok.value)
            self._get_next_token()
        self._match(TokenKind.OPERATOR, ')')
        return PrototypeAST(name, argnames)

    # external ::= 'extern' prototype
    def _parse_external(self):
        self._get_next_token()  # consume 'extern'
        return self._parse_prototype()

    # definition ::= 'def' prototype expression
    def _parse_definition(self):
        self._get_next_token()  # consume 'def'
        proto = self._parse_prototype()
        expr = self._parse_expression()
        return FunctionAST(proto, expr)

    # toplevel ::= expression
    def _parse_toplevel_expression(self):
        expr = self._parse_expression()
        # Anonymous function
        return FunctionAST(PrototypeAST('', []), expr)


#---- Some unit tests ----#

import unittest

class TestLexer(unittest.TestCase):
    def _assert_toks(self, toks, kinds):
        """Assert that the list of toks has the given kinds."""
        self.assertEqual([t.kind.name for t in toks], kinds)

    def test_lexer_simple_tokens_and_values(self):
        l = Lexer('a+1')
        toks = list(l.tokens())
        self.assertEqual(toks[0], Token(TokenKind.IDENTIFIER, 'a'))
        self.assertEqual(toks[1], Token(TokenKind.OPERATOR, '+'))
        self.assertEqual(toks[2], Token(TokenKind.NUMBER, '1'))
        self.assertEqual(toks[3], Token(TokenKind.EOF, ''))

        l = Lexer('.1519')
        toks = list(l.tokens())
        self.assertEqual(toks[0], Token(TokenKind.NUMBER, '.1519'))

    def test_token_kinds(self):
        l = Lexer('10.1 def der extern foo (')
        self._assert_toks(
            list(l.tokens()),
            ['NUMBER', 'DEF', 'IDENTIFIER', 'EXTERN', 'IDENTIFIER',
             'OPERATOR', 'EOF'])

        l = Lexer('+- 1 2 22 22.4 a b2 C3d')
        self._assert_toks(
            list(l.tokens()),
            ['OPERATOR', 'OPERATOR', 'NUMBER', 'NUMBER', 'NUMBER', 'NUMBER',
             'IDENTIFIER', 'IDENTIFIER', 'IDENTIFIER', 'EOF'])

    def test_skip_whitespace_comments(self):
        l = Lexer('''
            def foo # this is a comment
            # another comment
            \t\t\t10
            ''')
        self._assert_toks(
            list(l.tokens()),
            ['DEF', 'IDENTIFIER', 'NUMBER', 'EOF'])


class TestParser(unittest.TestCase):
    def _flatten(self, ast):
        """Test helper - flattens the AST into a sexpr-like nested list."""
        if isinstance(ast, NumberExprAST):
            return ['Number', ast.val]
        elif isinstance(ast, VariableExprAST):
            return ['Variable', ast.name]
        elif isinstance(ast, BinaryExprAST):
            return ['Binop', ast.op,
                    self._flatten(ast.lhs), self._flatten(ast.rhs)]
        elif isinstance(ast, CallExprAST):
            args = [self._flatten(arg) for arg in ast.args]
            return ['Call', ast.callee, args]
        elif isinstance(ast, PrototypeAST):
            return ['Proto', ast.name, ' '.join(ast.argnames)]
        elif isinstance(ast, FunctionAST):
            return ['Function',
                    self._flatten(ast.proto), self._flatten(ast.body)]
        else:
            raise TypeError('unknown type in _flatten: {0}'.format(type(ast)))

    def _assert_body(self, toplevel, expected):
        """Assert the flattened body of the given toplevel function"""
        self.assertIsInstance(toplevel, FunctionAST)
        self.assertEqual(self._flatten(toplevel.body), expected)

    def test_basic(self):
        ast = Parser().parse_toplevel('2')
        self.assertIsInstance(ast, FunctionAST)
        self.assertIsInstance(ast.body, NumberExprAST)
        self.assertEqual(ast.body.val, '2')

    def test_basic_with_flattening(self):
        ast = Parser().parse_toplevel('2')
        self._assert_body(ast, ['Number', '2'])

        ast = Parser().parse_toplevel('foobar')
        self._assert_body(ast, ['Variable', 'foobar'])

    def test_expr_singleprec(self):
        ast = Parser().parse_toplevel('2+ 3-4')
        self._assert_body(ast,
            ['Binop',
                '-', ['Binop', '+', ['Number', '2'], ['Number', '3']],
                ['Number', '4']])

    def test_expr_multiprec(self):
        ast = Parser().parse_toplevel('2+3*4-9')
        self._assert_body(ast,
            ['Binop', '-',
                ['Binop', '+',
                    ['Number', '2'],
                    ['Binop', '*', ['Number', '3'], ['Number', '4']]],
                ['Number', '9']])

    def test_expr_parens(self):
        ast = Parser().parse_toplevel('2*(3-4)*7')
        self._assert_body(ast,
            ['Binop', '*',
                ['Binop', '*',
                    ['Number', '2'],
                    ['Binop', '-', ['Number', '3'], ['Number', '4']]],
                ['Number', '7']])

    def test_externals(self):
        ast = Parser().parse_toplevel('extern sin(arg)')
        self.assertEqual(self._flatten(ast), ['Proto', 'sin', 'arg'])

        ast = Parser().parse_toplevel('extern Foobar(nom denom abom)')
        self.assertEqual(self._flatten(ast),
            ['Proto', 'Foobar', 'nom denom abom'])

    def test_funcdef(self):
        ast = Parser().parse_toplevel('def foo(x) 1 + bar(x)')
        self.assertEqual(self._flatten(ast),
            ['Function', ['Proto', 'foo', 'x'],
                ['Binop', '+',
                    ['Number', '1'],
                    ['Call', 'bar', [['Variable', 'x']]]]])


if __name__ == '__main__':
    # We just have the lexer and parser here, no code generation yet. This is
    # just a simple way to parse Kaleidoscope expressions and dump the AST.
    p = Parser()
    print(p.parse_toplevel('def bina(a b) a + b').dump())
