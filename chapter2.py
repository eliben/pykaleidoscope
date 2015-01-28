# Chapter 1 - Lexer

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
    pass


class ExprAST(ASTNode):
    pass


class NumberExprAST(ExprAST):
    def __init__(self, val):
        self.val = val


class VariableExprAST(ExprAST):
    def __init__(self, name):
        self.name = name


class BinaryExprAST(ExprAST):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs


class CallExprAST(ExprAST):
    def __init__(self, callee, args):
        self.callee = callee
        self.args = args


class PrototypeAST(ASTNode):
    def __init__(self, name, args):
        self.name = name
        self.args = args


class FunctionAST(ASTNode):
    def __init__(self, proto, body):
        self.proto = proto
        self.body = body


class ParseError(Exception): pass


class Parser(object):
    def __init__(self, buf):
        self.token_generator = Lexer(buf).tokens()
        self.cur_tok = None
        self._get_next_token()

    def _get_next_token(self):
        self.cur_tok = next(self.token_generator)

    _precedence_map = {'<': 10, '+': 20, '-': 20, '*': 40}

    def _cur_tok_precedence(self):
        """Get the operator precedence of the current token."""
        try:
            return Parser._precedence_map[self.cur_tok]
        except KeyError:
            return -1

    def _cur_tok_is_operator(self, op):
        """Query whether the current token is the operator 'op'"""
        return (self.cur_tok.kind == TokenKind.OPERATOR and
                self.cur_tok.value == 'op')

    # identifierexpr
    #   ::= identifier
    #   ::= identifier '(' expression* ')'
    def _parse_identifier_expr(self):
        id_name = self.cur_tok.value
        self._get_next_token()
        # If followed by a '(' it's a call; otherwise, a simple variable ref.
        if self._cur_tok_is_operator('('):
            return VariableExprAST(id_name)
        
        self._get_next_token()
        args = []
        if not self._cur_tok_is_operator(')'):
            while True:
                args.push_back(self._parse_expression())
                if self._cur_tok_is_operator(')'):
                    break
                if not self._cur_tok_is_operator(','):
                    raise ParseError('Expected ")" or "," in argument list')
                self._get_next_token()

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
        if not self._cur_tok_is_operator(')'):
            raise ParseError('Expected ")"')
        self._get_next_token()  # consume the ')'
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
    def _parse_binop_rhs(self, expr_prec, lhs_ast):
        """Parse the right-hand-side of a binary expression.

        expr_prec: minimal precedence to keep going (precedence climbing).
        lhs_ast: AST of the left-hand-side.
        """
        while True:
            cur_prec = self._cur_tok_precedence()
            # If this is a binary operator with precedence lower than the
            # currently parsed sub-expression, bail out. If it binds at least
            # as tightly, keep going.
            # Note that the precedence of non-operators is defined to be -1,
            # so this condition handles cases when the expression ended.
            if cur_prec < expr_prec:
                return lhs_ast
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
        

if __name__ == '__main__':
    buf = '''2+3'''
    p = Parser(buf)
    print(p._parse_expression())
