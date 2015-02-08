# Chapter 7 - Mutable variables

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
    FOR = -10
    IN = -11
    BINARY = -12
    UNARY = -13
    VAR = -14


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
            'for':      TokenKind.FOR,
            'in':       TokenKind.IN,
            'binary':   TokenKind.BINARY,
            'unary':    TokenKind.UNARY,
            'var':      TokenKind.VAR,
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


class VarExprAST(ExprAST):
    def __init__(self, vars, body):
        # vars is a sequence of (name, init) pairs
        self.vars = vars
        self.body = body

    def dump(self, indent=0):
        prefix = ' ' * indent
        s = '{0}{1}\n'.format(prefix, self.__class__.__name__)
        for name, init in self.vars:
            s += '{0} {1}'.format(prefix, name)
            if init is None:
                s += '\n'
            else:
                s += '=\n' + init.dump(indent+2) + '\n'
        s += '{0} Body:\n'.format(prefix)
        s += self.body.dump(indent + 2)
        return s


class UnaryExprAST(ExprAST):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def dump(self, indent=0):
        s = '{0}{1}[{2}]\n'.format(
            ' ' * indent, self.__class__.__name__, self.op)
        s += self.operand.dump(indent + 2)
        return s


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


class ForExprAST(ExprAST):
    def __init__(self, id_name, start_expr, end_expr, step_expr, body):
        self.id_name = id_name
        self.start_expr = start_expr
        self.end_expr = end_expr
        self.step_expr = step_expr
        self.body = body

    def dump(self, indent=0):
        prefix = ' ' * indent
        s = '{0}{1}\n'.format(prefix, self.__class__.__name__)
        s += '{0} Start [{1}]:\n{2}\n'.format(
            prefix, self.id_name, self.start_expr.dump(indent + 2))
        s += '{0} End:\n{1}\n'.format(
            prefix, self.end_expr.dump(indent + 2))
        s += '{0} Step:\n{1}\n'.format(
            prefix, self.step_expr.dump(indent + 2))
        s += '{0} Body:\n{1}\n'.format(
            prefix, self.body.dump(indent + 2))
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
    def __init__(self, name, argnames, isoperator=False, prec=0):
        self.name = name
        self.argnames = argnames
        self.isoperator = isoperator
        self.prec = prec

    def is_unary_op(self):
        return self.isoperator and len(self.argnames) == 1

    def is_binary_op(self):
        return self.isoperator and len(self.argnames) == 2

    def get_op_name(self):
        assert self.isoperator
        return self.name[-1]

    def dump(self, indent=0):
        s = '{0}{1} {2}({3})'.format(
            ' ' * indent, self.__class__.__name__, self.name,
            ', '.join(self.argnames))
        if self.isoperator:
            s += '[operator with prec={0}]'.format(self.prec)
        return s


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

    _precedence_map = {'=': 2, '<': 10, '+': 20, '-': 20, '*': 40}

    def _cur_tok_precedence(self):
        """Get the operator precedence of the current token."""
        try:
            return self._precedence_map[self.cur_tok.value]
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
    #   ::= ifexpr
    #   ::= forexpr
    def _parse_primary(self):
        if self.cur_tok.kind == TokenKind.IDENTIFIER:
            return self._parse_identifier_expr()
        elif self.cur_tok.kind == TokenKind.NUMBER:
            return self._parse_number_expr()
        elif self._cur_tok_is_operator('('):
            return self._parse_paren_expr()
        elif self.cur_tok.kind == TokenKind.IF:
            return self._parse_if_expr()
        elif self.cur_tok.kind == TokenKind.FOR:
            return self._parse_for_expr()
        elif self.cur_tok.kind == TokenKind.VAR:
            return self._parse_var_expr()
        else:
            raise ParseError('Unknown token when expecting an expression')

    # ifexpr ::= 'if' expression 'then' expression 'else' expression
    def _parse_if_expr(self):
        self._get_next_token()  # consume the 'if'
        cond_expr = self._parse_expression()
        self._match(TokenKind.THEN)
        then_expr = self._parse_expression()
        self._match(TokenKind.ELSE)
        else_expr = self._parse_expression()
        return IfExprAST(cond_expr, then_expr, else_expr)

    # forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expr
    def _parse_for_expr(self):
        self._get_next_token()  # consume the 'for'
        id_name = self.cur_tok.value
        self._match(TokenKind.IDENTIFIER)
        self._match(TokenKind.OPERATOR, '=')
        start_expr = self._parse_expression()
        self._match(TokenKind.OPERATOR, ',')
        end_expr = self._parse_expression()

        # The step part is optional
        if self._cur_tok_is_operator(','):
            self._get_next_token()
            step_expr = self._parse_expression()
        else:
            step_expr = None
        self._match(TokenKind.IN)
        body = self._parse_expression()
        return ForExprAST(id_name, start_expr, end_expr, step_expr, body)

    # varexpr ::= 'var' identifier ('=' expr)?
    #                   (',' identifier ('=' expr)?)* 'in' expr
    def _parse_var_expr(self):
        self._get_next_token()  # consume the 'var'
        vars = []

        # At least one variable name is required
        if self.cur_tok.kind != TokenKind.IDENTIFIER:
            raise ParseError('expected identifier after "var"')
        while True:
            name = self.cur_tok.value
            self._get_next_token()  # consume the identifier

            # Parse the optional initializer
            if self._cur_tok_is_operator('='):
                self._get_next_token()  # consume the '='
                init = self._parse_expression()
            else:
                init = None
            vars.append((name, init))

            # If there are no more vars in this declaration, we're done.
            if not self._cur_tok_is_operator(','):
                break
            self._get_next_token()  # consume the ','
            if self.cur_tok.kind != TokenKind.IDENTIFIER:
                raise ParseError('expected identifier in "var" after ","')

        self._match(TokenKind.IN)
        body = self._parse_expression()
        return VarExprAST(vars, body)

    # unary
    #   ::= primary
    #   ::= <op> unary
    def _parse_unary(self):
        # no unary operator before a primary
        if (not self.cur_tok.kind == TokenKind.OPERATOR or
            self.cur_tok.value in ('(', ',')):
            return self._parse_primary()

        # unary operator
        op = self.cur_tok.value
        self._get_next_token()
        return UnaryExprAST(op, self._parse_unary())

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
            rhs = self._parse_unary()

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
        lhs = self._parse_unary()
        # Start with precedence 0 because we want to bind any operator to the
        # expression at this point.
        return self._parse_binop_rhs(0, lhs)

    # prototype
    #   ::= id '(' id* ')'
    #   ::= 'binary' LETTER number? '(' id id ')'
    def _parse_prototype(self):
        prec = 30
        if self.cur_tok.kind == TokenKind.IDENTIFIER:
            name = self.cur_tok.value
            self._get_next_token()
        elif self.cur_tok.kind == TokenKind.UNARY:
            self._get_next_token()
            if self.cur_tok.kind != TokenKind.OPERATOR:
                raise ParseError('Expected operator after "unary"')
            name = 'unary{0}'.format(self.cur_tok.value)
            self._get_next_token()
        elif self.cur_tok.kind == TokenKind.BINARY:
            self._get_next_token()
            if self.cur_tok.kind != TokenKind.OPERATOR:
                raise ParseError('Expected operator after "binary"')
            name = 'binary{0}'.format(self.cur_tok.value)
            self._get_next_token()

            # Try to parse precedence
            if self.cur_tok.kind == TokenKind.NUMBER:
                prec = int(self.cur_tok.value)
                if not (0 < prec < 101):
                    raise ParseError('Invalid precedence', prec)
                self._get_next_token()

            # Add the new operator to our precedence table so we can properly
            # parse it.
            self._precedence_map[name[-1]] = prec

        self._match(TokenKind.OPERATOR, '(')
        argnames = []
        while self.cur_tok.kind == TokenKind.IDENTIFIER:
            argnames.append(self.cur_tok.value)
            self._get_next_token()
        self._match(TokenKind.OPERATOR, ')')

        if name.startswith('binary') and len(argnames) != 2:
            raise ParseError('Expected binary operator to have 2 operands')
        elif name.startswith('unary') and len(argnames) != 1:
            raise ParseError('Expected unary operator to have one operand')

        return PrototypeAST(
            name, argnames, name.startswith(('unary', 'binary')), prec)

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
        # names to ir.Value which represents the var's address (alloca).
        self.func_symtab = {}

    def generate_code(self, node):
        assert isinstance(node, (PrototypeAST, FunctionAST))
        return self._codegen(node)

    def _create_entry_block_alloca(self, name):
        """Create an alloca in the entry BB of the current function."""
        builder = ir.IRBuilder()
        builder.position_at_start(self.builder.function.entry_basic_block)
        return builder.alloca(ir.DoubleType(), size=None, name=name)

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
        var_addr = self.func_symtab[node.name]
        return self.builder.load(var_addr, node.name)

    def _codegen_UnaryExprAST(self, node):
        operand = self._codegen(node.operand)
        func = self.module.globals.get('unary{0}'.format(node.op))
        return self.builder.call(func, [operand], 'unop')

    def _codegen_BinaryExprAST(self, node):
        # Assignment is handled specially because it doesn't follow the general
        # recipe of binary ops.
        if node.op == '=':
            if not isinstance(node.lhs, VariableExprAST):
                raise CodegenError('lhs of "=" must be a variable')
            var_addr = self.func_symtab[node.lhs.name]
            rhs_val = self._codegen(node.rhs)
            self.builder.store(rhs_val, var_addr)
            return rhs_val

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
            # Note one of predefined operator, so it must be a user-defined one.
            # Emit a call to it.
            func = self.module.globals.get('binary{0}'.format(node.op))
            return self.builder.call(func, [lhs, rhs], 'binop')

    def _codegen_IfExprAST(self, node):
        # Emit comparison value
        cond_val = self._codegen(node.cond_expr)
        cmp = self.builder.fcmp_ordered(
            '!=', cond_val, self.builder.constant(ir.DoubleType(), 0.0))

        # Create basic blocks to express the control flow, with a conditional
        # branch to either then_bb or else_bb depending on cmp. else_bb and
        # merge_bb are not yet attached to the function's list of BBs because
        # if a nested IfExpr is generated we want to have a reasonably nested
        # order of BBs generated into the function.
        then_bb = self.builder.function.append_basic_block('then')
        else_bb = ir.Block(self.builder.function, 'else')
        merge_bb = ir.Block(self.builder.function, 'ifcont')
        self.builder.cbranch(cmp, then_bb, else_bb)

        # Emit the 'then' part
        self.builder.position_at_start(then_bb)
        then_val = self._codegen(node.then_expr)
        self.builder.branch(merge_bb)

        # Emission of then_val could have modified the current basic block. To
        # properly set up the PHI, remember which block the 'then' part ends in.
        then_bb = self.builder.block

        # Emit the 'else' part
        self.builder.function.basic_blocks.append(else_bb)
        self.builder.position_at_start(else_bb)
        else_val = self._codegen(node.else_expr)

        # Emission of else_val could have modified the current basic block.
        else_bb = self.builder.block
        self.builder.branch(merge_bb)

        # Emit the merge ('ifcnt') block
        self.builder.function.basic_blocks.append(merge_bb)
        self.builder.position_at_start(merge_bb)
        phi = self.builder.phi(ir.DoubleType(), 'iftmp')
        phi.add_incoming(then_val, then_bb)
        phi.add_incoming(else_val, else_bb)
        return phi

    def _codegen_ForExprAST(self, node):
        # Output this as:
        #   var = alloca double
        #   ...
        #   start = startexpr
        #   store start -> var
        #   goto loop
        # loop:
        #   ...
        #   bodyexpr
        #   ...
        # loopend:
        #   step = stepexpr
        #   endcond = endexpr
        #   curvar = load var
        #   nextvariable = curvar + step
        #   store nextvar -> var
        #   br endcond, loop, afterloop
        # afterloop:

        # Create an alloca for the induction var. Save and restore location of
        # our builder because _create_entry_block_alloca may modify it (llvmlite
        # issue #44).
        saved_block = self.builder.block
        var_addr = self._create_entry_block_alloca(node.id_name)
        self.builder.position_at_end(saved_block)

        # Emit the start expr first, without the variable in scope. Store it
        # into the var.
        start_val = self._codegen(node.start_expr)
        self.builder.store(start_val, var_addr)
        loop_bb = self.builder.function.append_basic_block('loop')

        # Insert an explicit fall through from the current block to loop_bb
        self.builder.branch(loop_bb)
        self.builder.position_at_start(loop_bb)

        # Within the loop, the variable now refers to our alloca slot. If it
        # shadows an existing variable, we'll have to restore, so save it now.
        old_var_addr = self.func_symtab.get(node.id_name)
        self.func_symtab[node.id_name] = var_addr

        # Emit the body of the loop. This, like any other expr, can change the
        # current BB. Note that we ignore the value computed by the body.
        body_val = self._codegen(node.body)

        # Compute the end condition
        endcond = self._codegen(node.end_expr)
        cmp = self.builder.fcmp_ordered(
            '!=', endcond, self.builder.constant(ir.DoubleType(), 0.0),
            'loopcond')

        if node.step_expr is None:
            stepval = self.builder.constant(ir.DoubleType(), 1.0)
        else:
            stepval = self._codegen(node.step_expr)
        cur_var = self.builder.load(var_addr, node.id_name)
        nextval = self.builder.fadd(cur_var, stepval, 'nextvar')
        self.builder.store(nextval, var_addr)

        # Create the 'after loop' block and insert it
        after_bb = self.builder.function.append_basic_block('afterloop')

        # Insert the conditional branch into the end of loop_end_bb
        self.builder.cbranch(cmp, loop_bb, after_bb)

        # New code will be inserted into after_bb
        self.builder.position_at_start(after_bb)

        # Restore the old var address if it was shadowed.
        if old_var_addr is not None:
            self.func_symtab[node.id_name] = old_var_addr
        else:
            del self.func_symtab[node.id_name]

        # The 'for' expression always returns 0
        return self.builder.constant(ir.DoubleType(), 0.0)

    def _codegen_VarExprAST(self, node):
        old_bindings = []

        for name, init in node.vars:
            # Emit the initializer before adding the variable to scope. This
            # prefents the initializer from referencing the variable itself.
            if init is not None:
                init_val = self._codegen(init)
            else:
                init_val = self.builder.constant(ir.DoubleType(), 0.0)

            # Create an alloca for the induction var and store the init value to
            # it. Save and restore location of our builder because
            # _create_entry_block_alloca may modify it (llvmlite issue #44).
            saved_block = self.builder.block
            var_addr = self._create_entry_block_alloca(name)
            self.builder.position_at_end(saved_block)
            self.builder.store(init_val, var_addr)

            # We're going to shadow this name in the symbol table now; remember
            # what to restore.
            old_bindings.append(self.func_symtab.get(name))
            self.func_symtab[name] = var_addr

        # Now all the vars are in scope. Codegen the body.
        body_val = self._codegen(node.body)

        # Restore the old bindings.
        for i, (name, _) in enumerate(node.vars):
            if old_bindings[i] is not None:
                self.func_symtab[name] = old_bindings[i]
            else:
                del self.func_symtab[name]

        return body_val

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

        # Add all arguments to the symbol table and create their allocas
        for i, arg in enumerate(func.args):
            arg.name = node.proto.argnames[i]
            alloca = self.builder.alloca(ir.DoubleType(), name=arg.name)
            self.builder.store(arg, alloca)
            self.func_symtab[arg.name] = alloca

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
        self.parser = Parser()
        self._add_builtins(self.codegen.module)

        self.target = llvm.Target.from_default_triple()

    def evaluate(self, codestr, optimize=True, llvmdump=False):
        """Evaluate code in codestr.

        Returns None for definitions and externs, and the evaluated expression
        value for toplevel expressions.
        """
        # Parse the given code and generate code from it
        ast = self.parser.parse_toplevel(codestr)
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

    def _add_builtins(self, module):
        # The C++ tutorial adds putchard() simply by defining it in the host C++
        # code, which is then accessible to the JIT. It doesn't work as simply
        # for us; but luckily it's very easy to define new "C level" functions
        # for our JITed code to use - just emit them as LLVM IR. This is what
        # this method does.

        # Add the declaration of putchar
        putchar_ty = ir.FunctionType(ir.IntType(32), [ir.IntType(32)])
        putchar = ir.Function(module, putchar_ty, 'putchar')

        # Add putchard
        putchard_ty = ir.FunctionType(ir.DoubleType(), [ir.DoubleType()])
        putchard = ir.Function(module, putchard_ty, 'putchard')
        irbuilder = ir.IRBuilder(putchard.append_basic_block('entry'))
        ival = irbuilder.fptoui(putchard.args[0], ir.IntType(32), 'intcast')
        irbuilder.call(putchar, [ival])
        irbuilder.ret(irbuilder.constant(ir.DoubleType(), 0))


#---- Some unit tests ----#

import unittest


class TestParser(unittest.TestCase):
    def _flatten(self, ast):
        """Test helper - flattens the AST into a sexpr-like nested list."""
        if isinstance(ast, NumberExprAST):
            return ['Number', ast.val]
        elif isinstance(ast, VariableExprAST):
            return ['Variable', ast.name]
        elif isinstance(ast, UnaryExprAST):
            return ['Unary', ast.op, self._flatten(ast.operand)]
        elif isinstance(ast, BinaryExprAST):
            return ['Binop', ast.op,
                    self._flatten(ast.lhs), self._flatten(ast.rhs)]
        elif isinstance(ast, VarExprAST):
            vars = [[name, self._flatten(init)] for name, init in ast.vars]
            return ['Var', vars, self._flatten(ast.body)]
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

    def test_assignment(self):
        p = Parser()
        ast = p.parse_toplevel('def text(x) x = 5')
        self._assert_body(ast,
            ['Binop', '=', ['Variable', 'x'], ['Number', '5']])

    def test_varexpr(self):
        p = Parser()
        ast = p.parse_toplevel('def foo(x y) var t = 1 in y')
        self._assert_body(ast,
             ['Var', [['t', ['Number', '1']]], ['Variable', 'y']])
        ast = p.parse_toplevel('def foo(x y) var t = x, p = y + 1 in y')
        self._assert_body(ast,
            ['Var',
                [['t', ['Variable', 'x']],
                 ['p', ['Binop', '+', ['Variable', 'y'], ['Number', '1']]]],
                ['Variable', 'y']])


class TestEvaluator(unittest.TestCase):
    def test_var_expr(self):
        e = KaleidoscopeEvaluator()
        e.evaluate('''
            def foo(x y z)
                var s1 = x + y, s2 = z + y in
                    s1 * s2
            ''')
        self.assertEqual(e.evaluate('foo(1, 2, 3)'), 15)

        e = KaleidoscopeEvaluator()
        e.evaluate('def binary : 1 (x y) y')
        e.evaluate('''
            def foo(step)
                var accum in
                    (for i = 0, i < 10, step in
                        accum = accum + i) : accum
            ''')
        # Note that Kaleidoscope's 'for' loop executes the last iteration even
        # when the condition is no longer fulfilled after the step is done.
        # 0 + 2 + 4 + 6 + 8 + 10
        self.assertEqual(e.evaluate('foo(2)'), 30)

    def test_nested_var_exprs(self):
        e = KaleidoscopeEvaluator()
        e.evaluate('''
            def foo(x y z)
                var s1 = x + y, s2 = z + y in
                    var s3 = s1 * s2 in
                        s3 * 100
            ''')
        self.assertEqual(e.evaluate('foo(1, 2, 3)'), 1500)

    def test_assignments(self):
        e = KaleidoscopeEvaluator()
        e.evaluate('def binary : 1 (x y) y')
        e.evaluate('''
            def foo(a b)
                var s, p, r in
                   s = a + b :
                   p = a * b :
                   r = s + 100 * p :
                   r
            ''')
        self.assertEqual(e.evaluate('foo(2, 3)'), 605)
        self.assertEqual(e.evaluate('foo(10, 20)'), 20030)


if __name__ == '__main__':
    # Evaluate some code.
    kalei = KaleidoscopeEvaluator()
    kalei.evaluate('def binary: 1 (x y) y')
    kalei.evaluate('''
        def foo(x y z)
            var s1 = x + y, s2 = z + y in
                s1 * s2
        ''')
    print(kalei.evaluate('foo(1, 2, 3)'))
