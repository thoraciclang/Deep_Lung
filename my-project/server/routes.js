const router = require('koa-router')()
const { spawn } = require('child-process-async')

router.get('/submit', async (ctx) => {
  const train = ctx.query.train
  console.log(train)
  // console.log(JSON.stringify(train))
  const { stdout, stderr, exitCode } = await spawn('python', ['survival/survival_LCCS.py', train])
  result = String(stdout)
  console.log(String(stderr))
  console.log(result)
  result=JSON.parse(result)

  ctx.body = result
})

router.get('/recom', async (ctx) =>{
  const train = ctx.query.train
  console.log(train.length)
  const { stdout, stderr, exitCode } = await spawn('python', ['survival/survival_recommendation.py', train])
  result = String(stdout)
  console.log(String(stderr))
  console.log(result)
  result=JSON.parse(result)
  ctx.body = result
})

module.exports = router