# child-process-async
The best way to override Node's [`child_process`](https://nodejs.org/api/child_process.html) module w/ Promises

`child-process-async` provides a **drop-in replacement** for the original
`child_process` functions, not just duplicate methods that return a Promise.
So when you call `exec(...)` we still return a `ChildProcess` instance, just
with `.then()` and `.catch()` added to it to make it promise-friendly.

## Install
```shell
npm install --save child-process-async
```

## Usage

```js
// OLD:
const { exec, spawn } = require('child_process');
// NEW:
const { exec, spawn } = require('child-process-async');
```

### `exec()`
```js
async function() {
  const { stdout, stderr } = await exec('ls -al');
  // OR:
  const child = await exec('ls -al', {});
  // do whatever you want with `child` here - it's a ChildProcess instance just
  // with promise-friendly `.then()` & `.catch()` functions added to it!
  child.stdin.write(...);
  child.stdout.pipe(...);
  child.stderr.on('data', (data) => ...);
  const { stdout, stderr } = await child;
}
```

### `spawn()`
```js
async function() {
  const { stdout, stderr, exitCode } = await spawn('ls', [ '-al' ]);
  // OR:
  const child = spawn('ls', [ '-al' ], {});
  // do whatever you want with `child` here - it's a ChildProcess instance just
  // with promise-friendly `.then()` & `.catch()` functions added to it!
  child.stdin.write(...);
  child.stdout.pipe(...);
  child.stderr.on('data', (data) => ...);
  const { stdout, stderr, exitCode } = await child;
}
```
