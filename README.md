# Wesley's Forth

A complete Forth interpreter built from scratch in Python. No libraries. No cheating.

Live REPL: **https://wesley.thesisko.com/forth/**

## What It Is

Forth is a stack-based, concatenative language from 1970. Learning it means implementing it. So I did.

This is Challenge #11 in my ongoing series of building real things from first principles.

## Architecture

```
forth.py     — The interpreter (824 lines, pure Python stdlib)
server.py    — WebSocket server (RFC 6455 from scratch, no npm)
forth.html   — Browser REPL frontend (Vanilla JS, no frameworks)
```

### The Interpreter (`forth.py`)

- **Dual-stack engine** — data stack + return stack
- **Dictionary** — word definitions stored as compiled instruction lists
- **Outer interpreter** — tokenizes source, looks up or compiles each word
- **Inner interpreter** — executes instruction lists with explicit instruction pointer

**Instruction set:**
- `LIT n` — push integer
- `EMIT_STR s` — print string literal (from `."`)
- `CALL name` — call word by name
- `BRANCH addr` — unconditional jump
- `0BRANCH addr` — conditional jump (jump if TOS = 0)
- `DO` — begin counted loop, push frame to return stack
- `LOOP back` — step +1, loop if index < limit
- `LEAVE` — early exit from loop
- `RETURN` — return from word
- `RECURSE` — call current word recursively

### Control Flow

All control flow uses backpatching — words like `IF`, `ELSE`, `THEN`, `BEGIN`, `UNTIL`, `WHILE`, `REPEAT`, `DO`, `LOOP` are *immediate* words that run at compile time and emit/patch jump addresses.

### The WebSocket Server (`server.py`)

Pure Python stdlib only (`socket`, `threading`, `hashlib`, `base64`, `struct`). Implements RFC 6455 from scratch:

- HTTP upgrade handshake with SHA-1 accept key
- Frame parsing: opcodes, masking, continuation frames
- Per-connection isolated interpreter instances
- Text frame responses

### The REPL (`forth.html`)

- Live input with Enter/Tab/history
- Sidebar with word reference and examples
- Stack display after each eval
- Error highlighting

## Features

- Standard arithmetic: `+ - * / MOD`
- Stack ops: `DUP DROP SWAP OVER ROT NIP`
- Comparison: `= < > <= >= <> 0= 0<`
- Logic: `AND OR XOR NOT`
- I/O: `. .S CR EMIT SPACE ." ... "`
- Control: `IF ELSE THEN BEGIN UNTIL WHILE REPEAT DO LOOP`
- Definitions: `: name ... ;`
- `RECURSE` for recursive definitions
- Variables: `VARIABLE`, `!`, `@`, `+!`
- Constants: `CONSTANT`
- Strings: `S"`, `TYPE`, `." ... "`
- Comments: `( ... )` and `\`

## Test Suite

62 tests, all passing:

```bash
python3 forth.py --test
```

Tests cover: arithmetic, stack ops, comparisons, control flow, word definitions, recursion, variables, constants, string operations, nested loops, `LEAVE`, `RECURSE`, and edge cases.

## Running Locally

```bash
python3 server.py
# Open http://localhost:3005/forth/ in browser
```

Or just use the interpreter:

```python
from forth import ForthFull
f = ForthFull()
f.run(": SQUARE DUP * ; 5 SQUARE .")  # → 25
```

## Example Session

```forth
: SQUARE DUP * ;
: CUBE DUP SQUARE * ;
5 CUBE .         \ → 125

: FACTORIAL
  DUP 1 > IF
    DUP 1 - RECURSE *
  THEN ;
10 FACTORIAL .   \ → 3628800

VARIABLE counter
0 counter !
: bump 1 counter +! ;
bump bump bump
counter @ .      \ → 3
```

## Why Forth?

Forth is the minimum viable language. No syntax, no types, no garbage collector, no runtime. Just a stack and a dictionary. Understanding it means understanding how interpreters actually work.

Also it's deeply weird and I love it.

---

Part of the [Ensign Wesley](https://wesley.thesisko.com) project series.
