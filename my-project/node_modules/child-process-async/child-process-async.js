/**
 * child-process-async.js
 * ----------------------
 *
 * @flow
 */

const child_process = require('child_process');

function exec(command, options) /*: ChildProcess & Promise<{ stdout:Buffer|string, stderr:Buffer|string }>*/ {
    let proc;
    const _promise = new Promise((resolve, reject) => {
        proc = child_process.exec(command, options, (err, stdout, stderr) =>
            err ? reject(err) : resolve({ stdout, stderr })
        );
    });
    Object.defineProperties(proc, {
        then: { value: _promise.then.bind(_promise) },
        catch: { value: _promise.catch.bind(_promise) },
    });
    return proc;
}

function spawn(command, args, options) /*: ChildProcess & Promise<{ stdout:Buffer|string|null, stderr:Buffer|string|null, exitCode:number }>*/ {
    const proc = child_process.spawn(command, args, options);
    const _promise = new Promise((resolve, reject) => {
        let stdout = (proc.stdout && proc.stdout.readable) ? Buffer.alloc(0) : null,
            stderr = (proc.stderr && proc.stderr.readable) ? Buffer.alloc(0) : null;
        if (Buffer.isBuffer(stdout)) {
            proc.stdout.on('data', (data) => stdout = Buffer.concat([ stdout, data ]));
        }
        if (Buffer.isBuffer(stderr)) {
            proc.stderr.on('data', (data) => stderr = Buffer.concat([ stderr, data ]));
        }
        proc.on('error', reject);
        proc.on('close', (exitCode) => resolve({ stdout, stderr, exitCode }));
    });
    Object.defineProperties(proc, {
        then: { value: _promise.then.bind(_promise) },
        catch: { value: _promise.catch.bind(_promise) },
    });
    return proc;
}

module.exports = Object.assign({}, child_process, { exec, spawn });
