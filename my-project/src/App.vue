<template>
  <div id="app">
    <div class="side-content">
      <component-view></component-view>
    </div>
    <div class="main-content">
      <cur-view></cur-view>
      <rec-view></rec-view>
    </div>
  </div>
</template>

<script>

import ComponentView from './components/ComponentView'
import CurView from './components/CurView'
import RecView from './components/RecView'

import { vaxios } from './request-common';
import {Curvis} from './js/curvis'
import {Recvis} from './js/recvis'

var curvis = Curvis()
var recvis = Recvis()

export default {
  name: 'App',
  components:{
    ComponentView,
    CurView,
    RecView
  },
  data (){
    return {
      feature: null
    }
  },
  watch: {
    
  },
  created(){
    Hub.$on('result', (res)=> {
      curvis.data(res.data).layout().update()
    })
    Hub.$on('re_result', (res)=> {
      console.log(res.data)
      recvis.data(res.data).layout().update()
    })
  },
  mounted() {
    // Compvis.container(d3.select('#component-view').append('svg'))
    curvis.container(d3.select('#cur-contain').append('svg'))
    curvis.init()
    recvis.container(d3.select('#rec-contain').append('svg'))
    recvis.init()
  }
}
</script>

