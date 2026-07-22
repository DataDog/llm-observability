# Testing local `dd-trace-js` changes

You do **not** need to modify your global Node.js environment or reinstall `dd-trace` in this examples directory to test local `dd-trace-js` changes.

The example supports `DD_TRACE_JS_PATH`, which makes it load your local checkout directly instead of resolving `dd-trace` from `node_modules`.

## Recommended setup

From this examples directory:

```sh
cd /Users/mehul.sonowal/dd/llm-observability/experiments/nodejs
cp .env.example .env
```

In `.env`, set:

```sh
DD_TRACE_JS_PATH=/Users/mehul.sonowal/go/src/github.com/DataDog/dd-trace-js
DD_SITE=datad0g.com
DD_LLMOBS_PROJECT_NAME=nodejs-experiments-examples
```

Add either:

```sh
DD_API_KEY=<your-datadog-api-key>
DD_APP_KEY=<your-datadog-application-key>
```

or use `dd-auth` for Datadog credentials at runtime.

## Run with `dd-auth`

This is the preferred staging command because it avoids storing Datadog keys in `.env`:

```sh
dd-auth --domain dd.datad0g.com -- env DD_SITE=datad0g.com npm run validate:dataset
```

The script will still load `DD_TRACE_JS_PATH` from `.env`.

## Run the dataset script against local `dd-trace-js`

```sh
dd-auth --domain dd.datad0g.com -- env DD_SITE=datad0g.com npm run dataset
```

or directly:

```sh
dd-auth --domain dd.datad0g.com -- env DD_SITE=datad0g.com node examples/00-dataset-operations.js
```

## Verify the local checkout is being used

Set this in `.env` or the shell:

```sh
DD_TRACE_JS_PATH=/Users/mehul.sonowal/go/src/github.com/DataDog/dd-trace-js
```

The helper in `examples/lib/env.js` will then do roughly:

```js
require(path.resolve(process.env.DD_TRACE_JS_PATH))
```

That means changes in your local checkout are used immediately by the example.

## When would you need to reinstall or link?

Usually, you do not.

Only use `npm link`, `npm install /path/to/dd-trace-js`, or a packed tarball if you specifically want to test package installation behavior. For SDK behavior changes, `DD_TRACE_JS_PATH` is simpler and less error-prone.

## Common gotcha

For staging:

- Use `DD_SITE=datad0g.com` for API calls.
- Generated UI links should point to `https://dd.datad0g.com/...`.

Do **not** set `DD_SITE=dd.datad0g.com`; the SDK API client builds `api.${DD_SITE}`, which would become `api.dd.datad0g.com`.
