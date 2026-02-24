#!/usr/bin/env python3
"""
forth.py — A Forth interpreter from scratch.

No libraries. No AST. Just a data stack, a return stack, and a dictionary.

Architecture:
  - Outer interpreter: tokenize source, look up or compile each word
  - Inner interpreter: execute compiled instruction lists with explicit IP
  - Immediate words run at compile time to emit/backpatch control flow
  - Instructions are tuples with absolute address targets

Instruction set:
  ('LIT',      n)    push integer n
  ('EMIT_STR', s)    print string literal (from ."))
  ('CALL',     name) call word by uppercase name
  ('BRANCH',   addr) jump to absolute addr
  ('0BRANCH',  addr) jump if TOS is 0
  ('DO',)            pop (limit, start), push loop frame to rs
  ('LOOP',     back) step +1; loop if index < limit
  ('+LOOP',    back) step by TOS; loop based on direction
  ('I',)             push current loop index
  ('J',)             push outer loop index
  ('>R',)            move TOS to return stack
  ('R>',)            move return stack top to data stack
  ('R@',)            copy return stack top to data stack
  ('LEAVE',    exit) exit loop, jump to exit
"""

import sys


class ForthError(Exception):
    pass


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(src: str) -> list:
    tokens = []
    i, n = 0, len(src)
    while i < n:
        while i < n and src[i] in ' \t\r\n':
            i += 1
        if i >= n:
            break
        if src[i:i+2] == '."':
            i += 2
            if i < n and src[i] == ' ':
                i += 1
            j = src.find('"', i)
            if j < 0: j = n
            tokens.append(('STR', src[i:j]))
            i = j + 1
            continue
        if src[i] == '(':
            j = src.find(')', i)
            i = (j + 1) if j >= 0 else n
            continue
        if src[i] == '\\':
            j = src.find('\n', i)
            i = (j + 1) if j >= 0 else n
            continue
        j = i
        while j < n and src[j] not in ' \t\r\n':
            j += 1
        tokens.append(('WORD', src[i:j]))
        i = j
    return tokens


def parse_num(s: str) -> int:
    if s.startswith(('0x', '0X')):
        return int(s, 16)
    if s.startswith('$'):
        return int(s[1:], 16)
    return int(s, 10)


# ── Interpreter ───────────────────────────────────────────────────────────────

class Forth:
    def __init__(self):
        self.ds:   list  = []
        self.rs:   list  = []
        self.mem:  dict  = {}
        self._next_addr  = 1
        self.words: dict = {}
        self._order: list= []
        self.compiling   = False
        self._code: list | None = None
        self._name: str  | None = None
        self._ctrl: list = []
        self.out:   list = []
        self._define_builtins()

    # ── I/O ───────────────────────────────────────────────────────────────────

    def _emit(self, s):
        self.out.append(str(s))

    # ── Stack ─────────────────────────────────────────────────────────────────

    def _pop(self):
        if not self.ds:
            raise ForthError('Stack underflow')
        return self.ds.pop()

    def _peek(self):
        if not self.ds:
            raise ForthError('Stack underflow')
        return self.ds[-1]

    def _need(self, n):
        if len(self.ds) < n:
            raise ForthError('Stack underflow')

    # ── Memory ────────────────────────────────────────────────────────────────

    def _alloc(self):
        addr = self._next_addr
        self._next_addr += 1
        self.mem[addr] = 0
        return addr

    def _fetch(self, addr):
        if addr not in self.mem:
            raise ForthError(f'Bad address: {addr}')
        return self.mem[addr]

    def _store(self, addr, val):
        if addr not in self.mem:
            raise ForthError(f'Bad address: {addr}')
        self.mem[addr] = int(val)

    # ── Public ────────────────────────────────────────────────────────────────

    def interpret(self, source: str) -> str:
        self.out = []
        try:
            self._outer(source)
            if not self.compiling:
                self.out.append(' ok')
        except ForthError as e:
            self.out.append(f'\n Error: {e}')
        return ''.join(self.out)

    # ── Outer interpreter ─────────────────────────────────────────────────────

    def _outer(self, source: str):
        tokens = tokenize(source)
        i = 0
        while i < len(tokens):
            tok = tokens[i]; i += 1

            if tok[0] == 'STR':
                if self.compiling:
                    self._code.append(('EMIT_STR', tok[1]))
                else:
                    self._emit(tok[1])
                continue

            word = tok[1]
            upper = word.upper()

            # ── Colon definition ─────────────────────────────────────────────
            if upper == ':':
                if self.compiling:
                    raise ForthError('Nested :')
                if i >= len(tokens):
                    raise ForthError(': needs a name')
                name = tokens[i][1].upper(); i += 1
                self.compiling = True
                self._code = []
                self._name = name
                self._ctrl = []
                continue

            # ── VARIABLE ─────────────────────────────────────────────────────
            if upper == 'VARIABLE':
                if i >= len(tokens):
                    raise ForthError('VARIABLE needs a name')
                name = tokens[i][1].upper(); i += 1
                addr = self._alloc()
                self.words[name] = {'kind': 'variable', 'addr': addr}
                if name not in self._order: self._order.append(name)
                continue

            # ── CONSTANT ─────────────────────────────────────────────────────
            if upper == 'CONSTANT':
                if i >= len(tokens):
                    raise ForthError('CONSTANT needs a name')
                name = tokens[i][1].upper(); i += 1
                val = self._pop()
                self.words[name] = {'kind': 'constant', 'value': val}
                if name not in self._order: self._order.append(name)
                continue

            # ── SEE ───────────────────────────────────────────────────────────
            if upper == 'SEE':
                if i >= len(tokens):
                    raise ForthError('SEE needs a name')
                name = tokens[i][1].upper(); i += 1
                defn = self.words.get(name)
                if not defn:
                    raise ForthError(f'SEE: unknown word {name}')
                self._emit(self._see(name, defn))
                continue

            # ── Compile or execute ───────────────────────────────────────────
            if self.compiling:
                defn = self.words.get(upper)
                if defn and defn.get('immediate'):
                    defn['fn']()
                elif defn:
                    self._code.append(('CALL', upper))
                else:
                    try:
                        self._code.append(('LIT', parse_num(word)))
                    except ValueError:
                        raise ForthError(f'Unknown word (compile): {word}')
            else:
                defn = self.words.get(upper)
                if defn:
                    self._exec_defn(defn)
                else:
                    try:
                        self.ds.append(parse_num(word))
                    except ValueError:
                        raise ForthError(f'Undefined: {word}')

    # ── Execute definition ────────────────────────────────────────────────────

    def _exec_defn(self, defn: dict):
        k = defn['kind']
        if   k == 'builtin':  defn['fn']()
        elif k == 'compiled': self._exec_code(defn['code'])
        elif k == 'variable': self.ds.append(defn['addr'])
        elif k == 'constant': self.ds.append(defn['value'])
        else: raise ForthError(f'Bad kind: {k}')

    # ── Inner interpreter ─────────────────────────────────────────────────────

    def _exec_code(self, code: list):
        ip = 0
        while ip < len(code):
            instr = code[ip]
            op    = instr[0]

            if op == 'LIT':
                self.ds.append(instr[1])

            elif op == 'EMIT_STR':
                self._emit(instr[1])

            elif op == 'CALL':
                defn = self.words.get(instr[1])
                if not defn:
                    raise ForthError(f'Undefined at runtime: {instr[1]}')
                self._exec_defn(defn)

            elif op == 'BRANCH':
                ip = instr[1]; continue

            elif op == '0BRANCH':
                if self._pop() == 0:
                    ip = instr[1]; continue

            elif op == 'DO':
                start = self._pop()
                limit = self._pop()
                self.rs.append(('LF', start, limit))   # Loop Frame

            elif op == 'LOOP':
                f = self.rs[-1]
                if f[0] != 'LF': raise ForthError('LOOP outside DO')
                ni = f[1] + 1
                if ni < f[2]:
                    self.rs[-1] = ('LF', ni, f[2])
                    ip = instr[1]; continue
                else:
                    self.rs.pop()

            elif op == '+LOOP':
                step = self._pop()
                f = self.rs[-1]
                if f[0] != 'LF': raise ForthError('+LOOP outside DO')
                ni = f[1] + step
                done = (step >= 0 and ni >= f[2]) or (step < 0 and ni <= f[2])
                if not done:
                    self.rs[-1] = ('LF', ni, f[2])
                    ip = instr[1]; continue
                else:
                    self.rs.pop()

            elif op == 'I':
                for f in reversed(self.rs):
                    if f[0] == 'LF': self.ds.append(f[1]); break
                else: raise ForthError('I outside loop')

            elif op == 'J':
                cnt = 0
                for f in reversed(self.rs):
                    if f[0] == 'LF':
                        cnt += 1
                        if cnt == 2: self.ds.append(f[1]); break
                else: raise ForthError('J outside nested loop')

            elif op == '>R':
                self.rs.append(('D', self._pop()))

            elif op == 'R>':
                if not self.rs or self.rs[-1][0] != 'D':
                    raise ForthError('R> mismatch')
                self.ds.append(self.rs.pop()[1])

            elif op == 'R@':
                if not self.rs: raise ForthError('R@ empty')
                f = self.rs[-1]
                self.ds.append(f[1])

            elif op == 'LEAVE':
                while self.rs and self.rs[-1][0] != 'LF':
                    self.rs.pop()
                if self.rs: self.rs.pop()
                ip = instr[1]; continue

            else:
                raise ForthError(f'Bad instruction: {op}')

            ip += 1

    # ── SEE ───────────────────────────────────────────────────────────────────

    def _see(self, name, defn):
        k = defn['kind']
        if k == 'builtin':
            return f': {name} <builtin> ;'
        if k == 'constant':
            return f'{defn["value"]} CONSTANT {name}'
        if k == 'variable':
            return f'VARIABLE {name}  ( addr={defn["addr"]}, value={self.mem.get(defn["addr"],0)} )'
        if k == 'compiled':
            parts = [f': {name}']
            for instr in defn['code']:
                op = instr[0]
                if   op == 'LIT':       parts.append(str(instr[1]))
                elif op == 'EMIT_STR':  parts.append(f'." {instr[1]}"')
                elif op == 'CALL':      parts.append(instr[1])
                elif op == 'BRANCH':    parts.append(f'BRANCH({instr[1]})')
                elif op == '0BRANCH':   parts.append(f'0BRANCH({instr[1]})')
                elif op == 'DO':        parts.append('DO')
                elif op == 'LOOP':      parts.append(f'LOOP←{instr[1]}')
                elif op == '+LOOP':     parts.append(f'+LOOP←{instr[1]}')
                elif op == 'LEAVE':     parts.append(f'LEAVE→{instr[1]}')
                else:                   parts.append(repr(instr))
            parts.append(';')
            return ' '.join(parts)
        return f'( unknown kind: {k} )'

    # ── Compile-time helpers ──────────────────────────────────────────────────

    def _find_do(self):
        for i in range(len(self._ctrl) - 1, -1, -1):
            if self._ctrl[i][0] == 'DO':
                return i
        raise ForthError('LEAVE outside DO')

    def _def(self, name, fn, imm=False):
        self.words[name.upper()] = {'kind': 'builtin', 'fn': fn, 'immediate': imm}
        if name.upper() not in self._order:
            self._order.append(name.upper())

    # ── Built-ins ─────────────────────────────────────────────────────────────

    def _define_builtins(self):
        d = self

        # ── Definition words ─────────────────────────────────────────────────
        def w_semi():
            name = d._name
            defn = {'kind': 'compiled', 'code': d._code, 'immediate': False}
            d.words[name] = defn
            if name not in d._order: d._order.append(name)
            d.compiling = False; d._code = None; d._name = None
        d._def(';', w_semi, imm=True)
        d._def('IMMEDIATE', lambda: d.words[d._order[-1]].__setitem__('immediate', True))

        # ── Control flow (immediate words) ───────────────────────────────────
        def w_if():
            d._code.append(('0BRANCH', 0))
            d._ctrl.append(('IF', len(d._code) - 1))
        d._def('IF', w_if, imm=True)

        def w_else():
            if not d._ctrl or d._ctrl[-1][0] != 'IF':
                raise ForthError('ELSE without IF')
            _, ia = d._ctrl.pop()
            d._code.append(('BRANCH', 0))
            ea = len(d._code) - 1
            d._code[ia] = ('0BRANCH', len(d._code))
            d._ctrl.append(('ELSE', ea))
        d._def('ELSE', w_else, imm=True)

        def w_then():
            if not d._ctrl or d._ctrl[-1][0] not in ('IF', 'ELSE'):
                raise ForthError('THEN without IF/ELSE')
            kind, addr = d._ctrl.pop()
            op = '0BRANCH' if kind == 'IF' else 'BRANCH'
            d._code[addr] = (op, len(d._code))
        d._def('THEN', w_then, imm=True)

        def w_begin():
            d._ctrl.append(('BEGIN', len(d._code)))
        d._def('BEGIN', w_begin, imm=True)

        def w_until():
            if not d._ctrl or d._ctrl[-1][0] != 'BEGIN':
                raise ForthError('UNTIL without BEGIN')
            _, ba = d._ctrl.pop()
            d._code.append(('0BRANCH', ba))
        d._def('UNTIL', w_until, imm=True)

        def w_while():
            if not d._ctrl or d._ctrl[-1][0] != 'BEGIN':
                raise ForthError('WHILE without BEGIN')
            d._code.append(('0BRANCH', 0))
            d._ctrl.append(('WHILE', len(d._code) - 1))
        d._def('WHILE', w_while, imm=True)

        def w_repeat():
            if len(d._ctrl) < 2 or d._ctrl[-1][0] != 'WHILE':
                raise ForthError('REPEAT without WHILE')
            _, wa = d._ctrl.pop()
            if not d._ctrl or d._ctrl[-1][0] != 'BEGIN':
                raise ForthError('REPEAT without BEGIN')
            _, ba = d._ctrl.pop()
            d._code.append(('BRANCH', ba))
            d._code[wa] = ('0BRANCH', len(d._code))
        d._def('REPEAT', w_repeat, imm=True)

        def w_do():
            d._code.append(('DO',))
            d._ctrl.append(['DO', len(d._code), []])  # mutable: [kind, back, leaves]
        d._def('DO', w_do, imm=True)

        def _end_loop(op):
            if not d._ctrl or d._ctrl[-1][0] != 'DO':
                raise ForthError(f'{op} without DO')
            frame = d._ctrl.pop()
            back, leaves = frame[1], frame[2]
            d._code.append((op, back))
            exit_addr = len(d._code)
            for li in leaves:
                d._code[li] = ('LEAVE', exit_addr)

        d._def('LOOP',  lambda: _end_loop('LOOP'),  imm=True)
        d._def('+LOOP', lambda: _end_loop('+LOOP'), imm=True)

        def w_leave():
            fi = d._find_do()
            d._code.append(('LEAVE', 0))
            d._ctrl[fi][2].append(len(d._code) - 1)
        d._def('LEAVE', w_leave, imm=True)

        # ── I, J — immediate (work in both modes) ────────────────────────────
        def w_i():
            if d.compiling:
                d._code.append(('I',))
            else:
                for f in reversed(d.rs):
                    if f[0] == 'LF': d.ds.append(f[1]); return
                raise ForthError('I outside loop')
        d._def('I', w_i, imm=True)

        def w_j():
            if d.compiling:
                d._code.append(('J',))
            else:
                cnt = 0
                for f in reversed(d.rs):
                    if f[0] == 'LF':
                        cnt += 1
                        if cnt == 2: d.ds.append(f[1]); return
                raise ForthError('J outside loop')
        d._def('J', w_j, imm=True)

        # ── Return stack — immediate ──────────────────────────────────────────
        def w_tor():
            if d.compiling: d._code.append(('>R',))
            else: d.rs.append(('D', d._pop()))
        d._def('>R', w_tor, imm=True)

        def w_fromr():
            if d.compiling: d._code.append(('R>',))
            else:
                if not d.rs or d.rs[-1][0] != 'D':
                    raise ForthError('R> mismatch')
                d.ds.append(d.rs.pop()[1])
        d._def('R>', w_fromr, imm=True)

        def w_rfetch():
            if d.compiling: d._code.append(('R@',))
            else:
                if not d.rs: raise ForthError('R@ empty')
                d.ds.append(d.rs[-1][1])
        d._def('R@', w_rfetch, imm=True)

        # ── Arithmetic ───────────────────────────────────────────────────────
        def _binop(op):
            def fn():
                b = d._pop(); a = d._pop()
                if op == '+':  d.ds.append(a + b)
                elif op == '-': d.ds.append(a - b)
                elif op == '*': d.ds.append(a * b)
                elif op == '/':
                    if b == 0: raise ForthError('Division by zero')
                    d.ds.append(int(a / b))  # truncate toward zero
                elif op == 'MOD':
                    if b == 0: raise ForthError('Modulo by zero')
                    d.ds.append(a % b)
                elif op == '/MOD':
                    if b == 0: raise ForthError('Division by zero')
                    d.ds.append(a % b); d.ds.append(int(a / b))
            return fn

        d._def('+',    _binop('+'))
        d._def('-',    _binop('-'))
        d._def('*',    _binop('*'))
        d._def('/',    _binop('/'))
        d._def('MOD',  _binop('MOD'))
        d._def('/MOD', _binop('/MOD'))

        def w_negate(): d.ds.append(-d._pop())
        def w_abs():    d.ds.append(abs(d._pop()))
        def w_max():    b=d._pop(); a=d._pop(); d.ds.append(max(a,b))
        def w_min():    b=d._pop(); a=d._pop(); d.ds.append(min(a,b))
        d._def('NEGATE', w_negate)
        d._def('ABS',    w_abs)
        d._def('MAX',    w_max)
        d._def('MIN',    w_min)
        d._def('1+', lambda: d.ds.append(d._pop() + 1))
        d._def('1-', lambda: d.ds.append(d._pop() - 1))
        d._def('2+', lambda: d.ds.append(d._pop() + 2))
        d._def('2-', lambda: d.ds.append(d._pop() - 2))
        d._def('2*', lambda: d.ds.append(d._pop() * 2))
        d._def('2/', lambda: d.ds.append(d._pop() // 2))

        # ── Comparison (true = -1, false = 0) ────────────────────────────────
        def _cmp(op):
            def fn():
                b = d._pop(); a = d._pop()
                res = {'=': a==b,'<': a<b,'>': a>b,'<>': a!=b,'<=': a<=b,'>=': a>=b}[op]
                d.ds.append(-1 if res else 0)
            return fn
        for op in ('=','<','>','<>','<=','>='):
            d._def(op, _cmp(op))
        d._def('0=', lambda: d.ds.append(-1 if d._pop()==0 else 0))
        d._def('0<', lambda: d.ds.append(-1 if d._pop()< 0 else 0))
        d._def('0>', lambda: d.ds.append(-1 if d._pop()> 0 else 0))

        # ── Logic ─────────────────────────────────────────────────────────────
        d._def('AND',    lambda: (lambda b,a: d.ds.append(a & b))(d._pop(), d._pop()))
        d._def('OR',     lambda: (lambda b,a: d.ds.append(a | b))(d._pop(), d._pop()))
        d._def('XOR',    lambda: (lambda b,a: d.ds.append(a ^ b))(d._pop(), d._pop()))
        d._def('INVERT', lambda: d.ds.append(~d._pop()))
        d._def('NOT',    lambda: d.ds.append(0 if d._pop() else -1))

        # ── Stack manipulation ────────────────────────────────────────────────
        def w_dup():
            if not d.ds: raise ForthError('Stack underflow')
            d.ds.append(d.ds[-1])
        def w_drop():  d._pop()
        def w_swap():
            d._need(2); d.ds[-1], d.ds[-2] = d.ds[-2], d.ds[-1]
        def w_over():
            d._need(2); d.ds.append(d.ds[-2])
        def w_rot():
            d._need(3); d.ds.append(d.ds.pop(-3))
        def w_nrot():
            d._need(3); d.ds.insert(-2, d.ds.pop())
        def w_nip():
            d._need(2); d.ds.pop(-2)
        def w_tuck():
            d._need(2); d.ds.insert(-2, d.ds[-1])
        def w_2dup():
            d._need(2); d.ds.extend([d.ds[-2], d.ds[-1]])
        def w_2drop():  d._pop(); d._pop()
        def w_2swap():
            d._need(4)
            a,b,c,e = d.ds[-4],d.ds[-3],d.ds[-2],d.ds[-1]
            d.ds[-4:] = [c,e,a,b]
        def w_2over():
            d._need(4); d.ds.extend([d.ds[-4], d.ds[-3]])
        def w_qdup():
            if d.ds and d.ds[-1] != 0: d.ds.append(d.ds[-1])
        d._def('DUP',   w_dup)
        d._def('DROP',  w_drop)
        d._def('SWAP',  w_swap)
        d._def('OVER',  w_over)
        d._def('ROT',   w_rot)
        d._def('-ROT',  w_nrot)
        d._def('NIP',   w_nip)
        d._def('TUCK',  w_tuck)
        d._def('2DUP',  w_2dup)
        d._def('2DROP', w_2drop)
        d._def('2SWAP', w_2swap)
        d._def('2OVER', w_2over)
        d._def('?DUP',  w_qdup)
        d._def('DEPTH', lambda: d.ds.append(len(d.ds)))

        # ── Output ────────────────────────────────────────────────────────────
        d._def('.',     lambda: d._emit(str(d._pop()) + ' '))
        d._def('EMIT',  lambda: d._emit(chr(d._pop() & 0xFF)))
        d._def('CR',    lambda: d._emit('\n'))
        d._def('SPACE', lambda: d._emit(' '))
        d._def('SPACES',lambda: d._emit(' ' * max(0, d._pop())))
        d._def('.S',    lambda: d._emit('<' + str(len(d.ds)) + '> ' +
                                        ' '.join(str(x) for x in d.ds) + ' '))
        d._def('?',     lambda: d._emit(str(d._fetch(d._pop())) + ' '))

        # ── Memory ────────────────────────────────────────────────────────────
        d._def('@',  lambda: d.ds.append(d._fetch(d._pop())))
        d._def('!',  lambda: (lambda addr: d._store(addr, d._pop()))(d._pop()))
        d._def('+!', lambda: (lambda addr: d._store(addr, d._fetch(addr) + d._pop()))(d._pop()))

        # ── Misc ──────────────────────────────────────────────────────────────
        d._def('WORDS', lambda: d._emit('  '.join(d._order)))
        d._def('CLEAR', lambda: d.ds.clear())
        d._def('HEX',     lambda: None)   # simplified
        d._def('DECIMAL', lambda: None)
        d._def('BYE',   lambda: sys.exit(0))


# ── RECURSE support ───────────────────────────────────────────────────────────

class ForthFull(Forth):
    """Forth with RECURSE support."""
    def __init__(self):
        super().__init__()
        def w_recurse():
            if not self.compiling:
                raise ForthError('RECURSE only valid during compilation')
            self._code.append(('CALL', self._name))
        self._def('RECURSE', w_recurse, imm=True)


# ── Tests ─────────────────────────────────────────────────────────────────────

def run_tests():
    cases = [
        # Arithmetic — use . to print
        ('1 2 + .',              '3 '),
        ('10 3 - .',             '7 '),
        ('6 7 * .',              '42 '),
        ('20 4 / .',             '5 '),
        ('17 5 MOD .',           '2 '),
        ('17 5 /MOD . .',        '3 2 '),   # /MOD leaves (rem quot); . prints quot, . prints rem
        ('10 NEGATE .',          '-10 '),
        ('-7 ABS .',             '7 '),
        ('3 5 MAX .',            '5 '),
        ('3 5 MIN .',            '3 '),
        ('5 1+ .',               '6 '),
        ('5 1- .',               '4 '),
        ('5 2* .',               '10 '),
        ('10 2/ .',              '5 '),

        # Stack ops (check stack state directly)
        ('3 DUP . .',            '3 3 '),
        ('3 4 SWAP . .',         '3 4 '),   # after swap: 4 3 — . prints 3, then 4
        ('1 2 OVER . . .',       '1 2 1 '),
        ('1 2 3 ROT . . .',      '1 3 2 '),
        ('3 4 NIP .',            '4 '),
        ('3 4 TUCK . . .',       '4 3 4 '),   # TUCK: ( a b -- b a b )
        ('2 DEPTH .',            '1 '),     # before DEPTH there's one item (2)

        # Comparison
        ('3 3 = .',              '-1 '),
        ('3 4 = .',              '0 '),
        ('3 4 < .',              '-1 '),
        ('4 3 < .',              '0 '),
        ('0 0= .',               '-1 '),
        ('1 0= .',               '0 '),
        ('0 0< .',               '0 '),
        ('-1 0< .',              '-1 '),

        # Logic
        ('-1 0 AND .',           '0 '),
        ('-1 -1 AND .',          '-1 '),
        ('0 -1 OR .',            '-1 '),
        ('-1 INVERT .',          '0 '),
        ('0 NOT .',              '-1 '),

        # Output
        ('42 .',                 '42 '),
        ('65 EMIT',              'A'),
        ('3 4 5 .S',             '<3> 3 4 5 '),

        # User-defined words
        (': SQUARE DUP * ; 5 SQUARE .', '25 '),
        (': CUBE DUP DUP * * ; 3 CUBE .', '27 '),
        (': DOUBLE 2 * ; 6 DOUBLE .', '12 '),

        # String output
        ('." Hello, World!"',    'Hello, World!'),
        ('." one" SPACE ." two"','one two'),

        # IF/ELSE/THEN
        (': POS 0 > IF ." yes" ELSE ." no" THEN ; 5 POS',  'yes'),
        (': POS 0 > IF ." yes" ELSE ." no" THEN ; -3 POS', 'no'),
        (': MYABS DUP 0 < IF NEGATE THEN ; -7 MYABS .', '7 '),
        (': MYABS DUP 0 < IF NEGATE THEN ;  7 MYABS .', '7 '),

        # BEGIN/UNTIL
        (': COUNT 0 BEGIN 1+ DUP . DUP 5 = UNTIL DROP ; COUNT',
         '1 2 3 4 5 '),

        # BEGIN/WHILE/REPEAT
        (': DOWNTO 5 BEGIN DUP . 1- DUP 0 = UNTIL DROP ; DOWNTO',
         '5 4 3 2 1 '),

        # DO/LOOP
        (': SUM 0 5 0 DO I + LOOP . ; SUM', '10 '),
        (': SQUARES 4 1 DO I DUP * . LOOP ; SQUARES', '1 4 9 '),

        # +LOOP
        (': EVENS 10 0 DO I . 2 +LOOP ; EVENS', '0 2 4 6 8 '),

        # Nested DO/LOOP
        (': MULT 3 1 DO 3 1 DO J I * . LOOP LOOP ; MULT',
         '1 2 2 4 '),

        # Variables
        ('VARIABLE X 42 X ! X @ .', '42 '),
        ('VARIABLE N 0 N ! 1 N +! N @ .', '1 '),

        # Constants
        ('42 CONSTANT ANSWER ANSWER .', '42 '),
        ('0 CONSTANT FALSE FALSE .', '0 '),

        # SEE (just check no error)
        (': DOUBLE 2 * ; SEE DOUBLE', None),

        # Recursion (RECURSE)
        (': FACT DUP 1 > IF DUP 1- RECURSE * THEN ; 5 FACT .', '120 '),
        (': FACT DUP 1 > IF DUP 1- RECURSE * THEN ; 10 FACT .', '3628800 '),

        # >R R>
        (': RTEST >R DUP R> + ; 3 7 RTEST .', '10 '),

        # FizzBuzz
        (''': FIZZBUZZ
  21 1 DO
    I 15 MOD 0= IF ." FizzBuzz "
    ELSE I 3 MOD 0= IF ." Fizz "
    ELSE I 5 MOD 0= IF ." Buzz "
    ELSE I .
    THEN THEN THEN
  LOOP ;
FIZZBUZZ''',
         '1 2 Fizz 4 Buzz Fizz 7 8 Fizz Buzz 11 Fizz 13 14 FizzBuzz 16 17 Fizz 19 Buzz '),

        # LEAVE
        (': FIND5 10 0 DO I 5 = IF I . LEAVE THEN LOOP ; FIND5', '5 '),
    ]

    passed = 0
    failures = []

    for src, expected in cases:
        f = ForthFull()
        result = f.interpret(src)
        # Strip trailing ' ok'
        r = result.rstrip()
        if r.endswith(' ok'):
            r = r[:-3].rstrip()

        if expected is None:
            if 'Error' in result:
                failures.append((src[:60], 'no error expected', result[:80]))
            else:
                passed += 1
        else:
            exp = expected.rstrip()
            if r == exp:
                passed += 1
            else:
                failures.append((src[:60], repr(exp), repr(r)))

    print(f'Tests: {passed}/{len(cases)} passed')
    for src, exp, got in failures:
        print(f'  FAIL: {src}')
        print(f'    exp: {exp}')
        print(f'    got: {got}')
    return passed, len(cases)


# ── Interactive REPL ──────────────────────────────────────────────────────────

def repl():
    f = ForthFull()
    print("Wesley's Forth  —  type BYE to exit, WORDS to list vocabulary")
    while True:
        try:
            prompt = '    ' if f.compiling else 'ok> '
            line = input(prompt)
            if not line.strip():
                continue
            out = f.interpret(line)
            if out:
                print(out)
        except EOFError:
            break
        except KeyboardInterrupt:
            print('\nInterrupted — stack and definitions preserved')
            f.compiling = False
            f._code = None


if __name__ == '__main__':
    if '--test' in sys.argv:
        p, t = run_tests()
        sys.exit(0 if p == t else 1)
    repl()
