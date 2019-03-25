// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'

import './style/style.css'

Vue.config.productionTip = false

window.Hub = new Vue();

/* eslint-disable no-new */
new Vue({
  el: '#app',
  data: {
    show: false
  },
  components: { App },
  template: '<App/>',
  http: {
    headers: {'Content-Type': 'application/x-www-form-urlencoded'}
  }
})

$(document).ready(function(){
  $("#app").bind("contextmenu",function(e){
    return false;
  });
});
