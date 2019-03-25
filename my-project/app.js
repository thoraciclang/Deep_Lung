const path = require('path')
const Koa = require('koa')
const convert = require('koa-convert')
const koaStatic = require('koa-static')
const bodyParser = require('koa-bodyparser')
const koaLogger = require('koa-logger')
// const session = require('koa-session')
const cors = require('koa2-cors')

const config = require('./server/config')
const routers = require('./server/routes')

const app = new Koa()

app.use(cors({
  credentials: true
}))

// 配置控制台日志中间件
app.use(convert(koaLogger()))

// 配置ctx.body解析中间件
app.use(bodyParser())

// 配置静态资源加载中间件
app.use(koaStatic(
  path.join(__dirname, '/data')
))
app.use(koaStatic(
  path.join(__dirname, '/dist')
))

const errorHandler = async (ctx, next) => {
  try {
    await next()
  } catch (err) {
    ctx.response.status = err.statusCode || err.status || 500
    ctx.response.body = {
      message: err.message
    }
    ctx.app.emit('error', err, ctx)
  }
}

app.on('error', function (err) {
  console.log('app.js onError, error: ', err.message)
  console.log(err)
})

app.use(errorHandler)

// 初始化路由中间件
app.use(routers.routes()).use(routers.allowedMethods())

// 监听启动端口
app.listen(config.port)
console.log(`the server starts at port ${config.port}`)
