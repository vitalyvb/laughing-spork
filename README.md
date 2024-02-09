# laughing-spork 

Programing language inspired by Lisp and YAML.

## Live demo
https://vitalyvb.github.io/eval/

Converted from Python to JavaScript. Python code had to be changed for the (mostly) successfull conversion.

Some features are not supported by the live versions, for example there's a parse error for YAML lists constructed with `[` and `]`.

Check the browser console for some (error) messages.

## Examples

Define a function `val`. The function computes expression `2 * 2` and returns the value. Call the function.

```yaml
---
- - define
  - val
  - - '*'
    - 2
    - 2
- val
```
Result: `[<Num: 4>]`

----

Flip-flop example with lambda and a closure
```yaml
---
- - define
  - flip-flop
  - - - lambda
      - - state
      - - lambda
        - []
        - - define
          - state
          - - '*'
            - state
            - -1
        - state
    - 1

- - flip-flop
- - flip-flop
- - flip-flop

```
Result: `[<Num: -1>, <Num: 1>, <Num: -1>]`

----

Lambda functions and function composition. 

```yaml
---
- - define
  - inverse
  - - lambda
    - - x
    - - begin
      - - '*'
        - x
        - -1
- - define
  - add2
  - - lambda
    - - x
    - - begin
      - - '+'
        - x
        - 2
- - define
  - compose
  - - lambda
    - - f
      - g
    - - lambda
      - - x
      - - f
        - - g
          - x
- - define
  - comp
  - - compose
    - inverse
    - add2
- - comp
  - 10
```
Result: `[<Num: -12>]`

----

A generator using [call-with-current-continuation](https://en.wikipedia.org/wiki/Call-with-current-continuation)

```yaml
---
- - define
  - iter
  - - lambda
    - - from
      - to
      - f
    - - if
      - - eq?
        - from
        - to
      - - f
        - from
      - - begin
        - - f
          - from
        - - iter
          - - '+'
            - from
            - 1
          - to
          - f

- - define
  - make-gen
  - - lambda
    - - from
      - to
    - - define
      - control-state

      - - lambda
        - - return
        - - iter
          - from
          - to

          - - lambda
            - - element
            - - define
              - return
              - - begin
                - - call/cc

                  - - lambda
                    - - resume-here
                    - - define
                      - control-state
                      - resume-here
                    - - return
                      - element

        - - return
          - "end"

    - - define
      - generator
      - - lambda
        - []
        - - call/cc
          - control-state

    - generator


- - define
  - gen
  - - make-gen
    - 0
    - 2

- - gen
- - gen
- - gen
- - gen

```

Result: `[<Num: 0>, <Num: 1>, <Num: 2>, <Str: 'end'>]`

----

Alternative syntax (not supported by the live demo version)
```yaml
---
- - define
  - fac'
  - - lambda
    - [ x, acc ]
    - [ if, [ 'eq?', x, 1 ], acc, [ fac', ['-',  x, 1 ], [ '*', acc, x ] ]]

- - define
  - fac
  - - lambda
    - [ x ]
    - [ fac', x, 1 ]

- [ fac, 20 ]
```
Result: `[<Num: 2432902008176640000>]`

## Built-in functions

Arithmetic and comparison
- `+`
- `*`
- `-`
- `/`
- eq?

List functions
- list
- first
- rest
- cons
- list?
- length
- empty?
- elem

Standard functions
- debug
- display
- sleep

Special functions:
- &apply
- &apply?
- &apply.exp
- &apply.args
- &apply.tolist
- &list.elist
- &list.eval
- &lambda
- &format
- &if

Very special functions:
- &rest
- begin
- call
- call/cc
- eval
- quote
- define
- defmacro
- callmacro

## Prelude macros

- lambda
- if
- defun
- cond

## Bugs

Yes.

## The name

The name was proposed by GitHub project name generator.

Seems appropriate.
