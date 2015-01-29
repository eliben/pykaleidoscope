# Chapter 5 - Extending the language: Control flow

from collections import namedtuple
from ctypes import CFUNCTYPE, c_double
from enum import Enum

import llvmlite.ir as ir
import llvmlite.binding as llvm

# Each token is a tuple of kind and value. kind is one of the enumeration values
# in TokenKind. value is the textual value of the token in the input.
class TokenKind(Enum):
    EOF = -1
    DEF = -2
    EXTERN = -3
    IDENTIFIER = -4
    NUMBER = -5
    OPERATOR = -6
    IF = -7
    THEN = -8
    ELSE = -9


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

        self._keyword_map = {
            'def':      TokenKind.DEF,
            'extern':   TokenKind.EXTERN,
            'if':       TokenKind.IF,
            'then':     TokenKind.THEN,
            'else':     TokenKind.ELSE,
        }

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
                if id_str in self._keyword_map:
                    yield Token(kind=self._keyword_map[id_str], value=id_str)
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


class IfExprAST(ExprAST):
    def __init__(self, cond_expr, then_expr, else_expr):
        self.cond_expr = cond_expr
        self.then_expr = then_expr
        self.else_expr = else_expr

    def dump(self, indent=0):
        prefix = ' ' * indent
        s = '{0}{1}\n'.format(prefix, self.__class__.__name__)
        s += '{0} Condition:\n{1}\n'.format(
            prefix, self.cond_expr.dump(indent + 2))
        s += '{0} Then:\n{1}\n'.format(
            prefix, self.then_expr.dump(indent + 2))
        s += '{0} Else:\n{1}'.format(
            prefix, self.else_expr.dump(indent + 2))
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

    _anonymous_function_counter = 0

    @classmethod
    def create_anonymous(klass, expr):
        """Create an anonymous function to hold an expression."""
        klass._anonymous_function_counter += 1
        return klass(
            PrototypeAST('_anon{0}'.format(klass._anonymous_function_counter),
                         []),
            expr)

    def is_anonymous(self):
        return self.proto.name.startswith('_anon')
        
    def dump(self, indent=0):
        s = '{0}{1}[{2}]\n'.format(
            ' ' * indent, self.__class__.__name__, self.proto.dump())
        s += self.body.dump(indent + 2) + '\n'
        return s


class ParseError(Exception): pass


class Parser(object):
    def __init__(self, buf):
        self.token_generator = Lexer(buf).tokens()
        self.cur_tok = None
        self._get_next_token()

    # toplevel ::= definition | external | expression | ';'
    def parse_toplevel(self):
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
    #   ::= ifexpr
    def _parse_primary(self):
        if self.cur_tok.kind == TokenKind.IDENTIFIER:
            return self._parse_identifier_expr()
        elif self.cur_tok.kind == TokenKind.NUMBER:
            return self._parse_number_expr()
        elif self._cur_tok_is_operator('('):
            return self._parse_paren_expr()
        elif self.cur_tok.kind == TokenKind.IF:
            return self._parse_if_expr()
        else:
            raise ParseError('Unknown token when expecting an expression')

    # ifexpr ::= 'if' expression 'then' expression 'else' expression
    def _parse_if_expr(self):
        self._get_next_token()  # consume the 'if'
        cond_expr = self._parse_expression()
        if self.cur_tok.kind != TokenKind.THEN:
            raise ParseError('Expected "then" in ifexpr')
        self._get_next_token()  # consume the 'then'
        then_expr = self._parse_expression()
        if self.cur_tok.kind != TokenKind.ELSE:
            raise ParseError('Expected "else" in ifexpr')
        self._get_next_token()  # consume the 'else'
        else_expr = self._parse_expression()
        return IfExprAST(cond_expr, then_expr, else_expr)
        
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
        if self.cur_tok.kind != TokenKind.IDENTIFIER:
            raise ParseError('Expected function name in prototype')
        name = self.cur_tok.value
        self._get_next_token()  # consume the name
        if not self._cur_tok_is_operator('('):
            raise ParseError('Expected "(" in prototype')
        self._get_next_token()  # consume '('
        argnames = []
        while self.cur_tok.kind == TokenKind.IDENTIFIER:
            argnames.append(self.cur_tok.value)
            self._get_next_token()
        if not self._cur_tok_is_operator(')'):
            raise ParseError('Expected ")" in prototype')
        self._get_next_token()  # consume ')'
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
        return FunctionAST.create_anonymous(expr)


class CodegenError(Exception): pass


class LLVMCodeGenerator(object):
    def __init__(self):
        """Initialize the code generator.

        This creates a new LLVM module into which code is generated. The
        generate_code() method can be called multiple times. It adds the code
        generated for this node into the module, and returns the IR value for
        the node.

        At any time, the current LLVM module being constructed can be obtained
        from the module attribute.
        """
        self.module = ir.Module()

        # Current IR builder.
        self.builder = None

        # Manages a symbol table while a function is being codegen'd. Maps var
        # names to ir.Value.
        self.func_symtab = {}

    def generate_code(self, node):
        assert isinstance(node, (PrototypeAST, FunctionAST))
        return self._codegen(node)

    def _codegen(self, node):
        """Node visitor. Dispathces upon node type.

        For AST node of class Foo, calls self._codegen_Foo. Each visitor is
        expected to return a llvmlite.ir.Value.
        """
        method = '_codegen_' + node.__class__.__name__
        return getattr(self, method)(node)

    def _codegen_NumberExprAST(self, node):
        return self.builder.constant(ir.DoubleType(), float(node.val))

    def _codegen_VariableExprAST(self, node):
        return self.func_symtab[node.name]

    def _codegen_BinaryExprAST(self, node):
        lhs = self._codegen(node.lhs)
        rhs = self._codegen(node.rhs)

        if node.op == '+':
            return self.builder.fadd(lhs, rhs, 'addtmp')
        elif node.op == '-':
            return self.builder.fsub(lhs, rhs, 'subtmp')
        elif node.op == '*':
            return self.builder.fmul(lhs, rhs, 'multmp')
        elif node.op == '<':
            cmp = self.builder.fcmp_unordered('<', lhs, rhs, 'cmptmp')
            return self.builder.uitofp(cmp, ir.DoubleType(), 'booltmp')
        else:
            raise CodegenError('Unknown binary operator', node.op)

    def _codegen_CallExprAST(self, node):
        callee_func = self.module.globals.get(node.callee, None)
        if callee_func is None or not isinstance(callee_func, ir.Function):
            raise CodegenError('Call to unknown function', node.callee)
        if len(callee_func.args) != len(node.args):
            raise CodegenError('Call argument length mismatch', node.callee)
        call_args = [self._codegen(arg) for arg in node.args]
        return self.builder.call(callee_func, call_args, 'calltmp')
        
    def _codegen_PrototypeAST(self, node):
        funcname = node.name
        # Create a function type
        func_ty = ir.FunctionType(ir.DoubleType(),
                                  [ir.DoubleType()] * len(node.argnames))

        # If a function with this name already exists in the module...
        if funcname in self.module.globals:
            # We only allow the case in which a declaration exists and now the
            # function is defined (or redeclared) with the same number of args.
            existing_func = self.module[funcname]
            if not isinstance(existing_func, ir.Function):
                raise CodegenError('Function/Global name collision', funcname)
            if not existing_func.is_declaration():
                raise CodegenError('Redifinition of {0}'.format(funcname))
            if len(existing_func.function_type.args) != len(func_ty.args):
                raise CodegenError(
                    'Redifinition with different number of arguments')
            func = self.module.globals[funcname]
        else:
            # Otherwise create a new function
            func = ir.Function(self.module, func_ty, funcname)
        # Set function argument names from AST
        for i, arg in enumerate(func.args):
            arg.name = node.argnames[i]
            self.func_symtab[arg.name] = arg
        return func

    def _codegen_FunctionAST(self, node):
        # Reset the symbol table. Prototype generation will pre-populate it with
        # function arguments.
        self.func_symtab = {}
        # Create the function skeleton from the prototype.
        func = self._codegen(node.proto)
        # Create the entry BB in the function and set the builder to it.
        bb_entry = func.append_basic_block('entry')
        self.builder = ir.IRBuilder(bb_entry)
        retval = self._codegen(node.body)
        self.builder.ret(retval)
        return func


class KaleidoscopeEvaluator(object):
    """Evaluator for Kaleidoscope expressions.

    Once an object is created, calls to evaluate() add new expressions to the
    module. Definitions (including externs) are only added into the IR - no
    JIT compilation occurs. When a toplevel expression is evaluated, the whole
    module is JITed and the result of the expression is returned.
    """
    def __init__(self):
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()

        self.codegen = LLVMCodeGenerator()

        self.target = llvm.Target.from_default_triple()
 
    def evaluate(self, codestr, optimize=True, llvmdump=False):
        """Evaluate code in codestr.

        Returns None for definitions and externs, and the evaluated expression
        value for toplevel expressions.
        """
        # Parse the given code and generate code from it
        ast = Parser(codestr).parse_toplevel()
        self.codegen.generate_code(ast)
        
        if llvmdump:
            print('======== Unoptimized LLVM IR')
            print(str(self.codegen.module))

        # If we're evaluating a definition or extern declaration, don't do
        # anything else. If we're evaluating an anonymous wrapper for a toplevel
        # expression, JIT-compile the module and run the function to get its
        # result.
        if not (isinstance(ast, FunctionAST) and ast.is_anonymous()):
            return None

        # Convert LLVM IR into in-memory representation
        llvmmod = llvm.parse_assembly(str(self.codegen.module))

        # Optimize the module
        if optimize:
            pmb = llvm.create_pass_manager_builder()
            pmb.opt_level = 2
            pm = llvm.create_module_pass_manager()
            pmb.populate(pm)
            pm.run(llvmmod)

            if llvmdump:
                print('======== Optimized LLVM IR')
                print(str(llvmmod))

        # Create a MCJIT execution engine to JIT-compile the module. Note that
        # ee takes ownership of target_machine, so it has to be recreated anew
        # each time we call create_mcjit_compiler.
        target_machine = self.target.create_target_machine()
        with llvm.create_mcjit_compiler(llvmmod, target_machine) as ee:
            ee.finalize_object()

            if llvmdump:
                print('======== Machine code')
                print(target_machine.emit_assembly(llvmmod))

            func = llvmmod.get_function(ast.proto.name)
            fptr = CFUNCTYPE(c_double)(ee.get_pointer_to_function(func))

            result = fptr()
            return result

#---- Some unit tests ----#

import unittest

class TestEvaluator(unittest.TestCase):
    def test_basic(self):
        e = KaleidoscopeEvaluator()
        self.assertEqual(e.evaluate('3'), 3.0)
        self.assertEqual(e.evaluate('3+3*4'), 15.0)

    def test_use_func(self):
        e = KaleidoscopeEvaluator()
        self.assertIsNone(e.evaluate('def adder(x y) x+y'))
        self.assertEqual(e.evaluate('adder(5, 4) + adder(3, 2)'), 14.0)

    def test_use_libc(self):
        e = KaleidoscopeEvaluator()
        self.assertIsNone(e.evaluate('extern ceil(x)'))
        self.assertEqual(e.evaluate('ceil(4.5)'), 5.0)
        self.assertIsNone(e.evaluate('extern floor(x)'))
        self.assertIsNone(e.evaluate('def cfadder(x) ceil(x) + floor(x)'))
        self.assertEqual(e.evaluate('cfadder(3.14)'), 7.0)


if __name__ == '__main__':
    buf = 'def foo(a b) a * if a < b then 3 + a else b'
    print(Parser(buf).parse_toplevel().dump())
    #kalei = KaleidoscopeEvaluator()
    #print(kalei.evaluate('def adder(a b) a + b'))
    #print(kalei.evaluate('def foo(x) (1+2+x)*(x+(1+2))'))
    #print(kalei.evaluate('foo(3)', optimize=True, llvmdump=True))
    #print(kalei.evaluate('foo(adder(3, 3)*4)', optimize=True, llvmdump=True))
