import axios from 'axios'
import config from '../server/config'

export const vaxios = axios.create({
  baseURL: process.env.NODE_ENV === 'production' ? `http://202.120.189.180:${config.port}/` : `http://localhost:${config.port}/`,
  withCredentials: true,
  maxContentLength: 20000,
})
