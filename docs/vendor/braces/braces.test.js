'use strict';

const test = require('node:test');
const assert = require('node:assert');
const braces = require('./');

const nested = n => '{'.repeat(n) + 'a' + '}'.repeat(n);

test('ordinary patterns still expand', () => {
  assert.deepStrictEqual(braces.expand('a{b,c}d'), ['abd', 'acd']);
  assert.deepStrictEqual(braces.expand('{1..3}'), ['1', '2', '3']);
  assert.deepStrictEqual(braces.expand('{a,{b,c}}'), ['a', 'b', 'c']);
  assert.deepStrictEqual(braces('a/{b,c}/d'), ['a/(b|c)/d']);
});

test('nesting up to the limit is accepted', () => {
  assert.doesNotThrow(() => braces.expand(nested(256)));
  assert.doesNotThrow(() => braces(nested(256)));
});

test('deeper nesting is a SyntaxError, not a stack overflow', () => {
  for (const depth of [257, 4900]) {
    assert.throws(() => braces.expand(nested(depth)), SyntaxError);
    assert.throws(() => braces(nested(depth)), SyntaxError);
  }
});
