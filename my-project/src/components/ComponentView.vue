<template>
  <div id = "component-view" class="view">
    <div class="view-header" id="profile-view-header">
      <label>Patient Information</label>
      <div id="pre-btn" @click="submit()">
        Predict
      </div>
      <div id="recom-btn" @click="recom()">
        Recommendation
      </div>
    </div>
    <!--category-->
    <div class='column'>
      <div style='position: relative'>
        <div class='feature_box'>
          <span class="feature_label1">Sex:</span>
          <div>
            <input type="text" id="pid-input_1">
          </div>
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(1)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_1" style="display:none; top: 38px; left:188px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list1" :key="idx" @click="selected(item,idx,1)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label1">Grade:</span>
          <input type="text" id="pid-input_2">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(2)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_2" style="display:none; top: 88px; left:188px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list2" :key="idx" @click="selected(item,idx,2)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label1">Race Recode:</span>
          <input type="text" id="pid-input_3">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(3)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_3" style="display:none; top: 138px; left:188px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list3" :key="idx" @click="selected(item,idx,3)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label1">Marital Status:</span>
          <input type="text" id="pid-input_4">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(4)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_4" style="display:none; top: 188px; left:188px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list4" :key="idx" @click="selected(item,idx,4)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label1">Stage:</span>
          <input type="text" id="pid-input_5">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(5)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_5" style="display:none; top: 238px; left:188px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list5" :key="idx" @click="selected(item,idx,5)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label1">T stage:</span>
          <input type="text" id="pid-input_6">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(6)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_6" style="display:none; top: 288px; left:188px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list6" :key="idx" @click="selected(item,idx,6)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label1">N stage:</span>
          <input type="text" id="pid-input_7">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(7)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_7" style="display:none; top: 338px; left:188px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list7" :key="idx" @click="selected(item,idx,7)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label1">M stage:</span>
          <input type="text" id="pid-input_8">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(8)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_8" style="display:none; top: 388px; left:188px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list8" :key="idx" @click="selected(item,idx,8)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label1">Primary Sit:</span>
          <input type="text" id="pid-input_9">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(9)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_9" style="display:none; top: 438px; left:188px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list9" :key="idx" @click="selected(item,idx,9)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label1">Histologic:</span>
          <input type="text" id="pid-input_10">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(10)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_10" style="display:none; top: 488px; left:188px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list10" :key="idx" @click="selected(item,idx,10)">
              <label v-text="item"></label>
          </div>
        </div>
      </div>
      <div style='position: relative'>
        <div class='feature_box'>
          <span class="feature_label2">RX Summ-Scope Reg LN Sur (2003+):</span>
          <input type="text" id="pid-input_11">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(11)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_11" style="display:none; top: 38px; left:267px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list11" :key="idx" @click="selected(item,idx,11)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label2">CS extension (2004+):</span>
          <input type="text" id="pid-input_12">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(12)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_12" style="display:none; top: 88px; left:267px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list12" :key="idx" @click="selected(item,idx,12)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label2">CS mets at dx (2004+):</span>
          <input type="text" id="pid-input_13">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(13)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_13" style="display:none; top: 138px; left:267px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list13" :key="idx" @click="selected(item,idx,13)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label2">Lung - Pleural/Elastic Layer Invasion (PL) by H and E or Elastic Stain:</span>
          <input type="text" id="pid-input_14">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(14)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_14" style="display:none; top: 188px; left:267px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list14" :key="idx" @click="selected(item,idx,14)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label2">Lung - Separate Tumor Nodules - Ipsilateral Lung:</span>
          <input type="text" id="pid-input_15">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(15)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_15" style="display:none; top: 238px; left:267px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list15" :key="idx" @click="selected(item,idx,15)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label2">Lung - Surgery to Primary Site (1988-2015):</span>
          <input type="text" id="pid-input_16">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(16)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_16" style="display:none; top: 288px; left:267px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list16" :key="idx" @click="selected(item,idx,16)">
              <label v-text="item"></label>
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label2">Lung - Surgery to Other Regional/Distant Sites (1998+):</span>
          <input type="text" id="pid-input_17">
          <div class="list--btn">  
            <button class="header--btn" @click="pidBtnClick(17)">
              <svg viewBox="0 0 24 24" class="header--icon">
                <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z" />
              </svg>
            </button>
          </div>
        </div>
        <div class="item_list" id="item_list_17" style="display:none; top: 338px; left:267px; z-index:10000">
          <div class="pid-item" v-for="(item, idx) in list17" :key="idx" @click="selected(item,idx,17)">
              <label v-text="item"></label>
          </div>
        </div>

        <!--numerical-->
        <div class='feature_box'>
          <span class="feature_label2">Age at diagnosis:</span>
          <input type="text" id="pid-input_18">
          <div class="list--btn">  
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label2">CS tumor size (2004+):</span>
          <input type="text" id="pid-input_19">
          <div class="list--btn">  
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label2">Regional nodes examined (1998+):</span>
          <input type="text" id="pid-input_20">
          <div class="list--btn">  
          </div>
        </div>
        <div class='feature_box'>
          <span class="feature_label2">Regional nodes positive (1998+):</span>
          <input type="text" id="pid-input_21">
          <div class="list--btn">    
          </div>
        </div>
      </div>
    </div>
  </div>

</template>

<script>
    import {vaxios} from '../request-common'
    export default {
      name: "ComponentView",
      data:function(){
        return {list1:['Female', 'Male'],
                list2:['Moderately','Poor','Undifine','Well'],
                list3:['White','African Americans','American Indian/Alaska Native','Asian or Pacific Islander'],
                list4:['Married','Un-married'],
                list5:['IA1','IA2','IA3','IB','IIA','IIB','IIIA','IIIB','IIIC','IVA','IVB'],
                list6:['T1a','T1b','T1c','T2a','T2b','T3','T4'],
                list7:['N0','N1','N2','N3'],
                list8:['M0','M1a','M1b','M1c'],
                list9:['Main bronchus/Carina/Hilum/Bronchus intermedius','Upper lobe, lung','Middle lobe, lung','Lower lobe, lung','Overlapping lesion of lung','Lung, NOS/Bronchus, NOS'],
                list10:['AD','SD'],
                list11:['None','Biopsy','Resection','System resection'],
                list12:['100','110','115','120','125','200','210','220','230','300','400','410','420','430','440','455','460','465','500','520','540','550','560','570','590','600','610','680','700','705','710','730','740','745','750','770','785'],
                list13:['0','15','16','17','18','20','21','23','24','25','26','30','32','33','36','37','38','40','41','42','43','51','52','53','70'],
                list14:['PL0','PL1','PL2','PL3'],
                list15:['No separate tumor nodules noted','Separate tumor nodules in ipsilateral lung, different lobe','Separate tumor nodules in ipsilateral lung, same lobe','Separate tumor nodules, ipsilateral lung, same and different lobe (010+020)'],
                list16:['Peumonectomy','Lobectomy','Sublobar','None'],
                list17:['None','Surgery To Distant Site Or Nodes','Surgery To Regional Site']}
      },
      watch:{

      },
      methods:{
        selected: function(item,idx,num) {
          d3.select('#pid-input_'+num).attr('value',item)
          d3.select('#pid-input_'+num).attr('key',idx)
          d3.select('#item_list_'+num).style('display', 'none');
        },
        pidBtnClick: function (num) {
            let display = d3.select('#item_list_'+num).style('display');
            if (display === 'none') {
                d3.select('#item_list_'+num).style('display', 'block');
                // this.pidList = this.pidVocab;
            }else  {
                d3.select('#item_list_'+num).style('display', 'none');
            }
        },
        submit: function() {
          var info = []
          for(var i=0; i<127; i++){
            info.push(0)
          }
          console.log(d3.select('#pid-input_18')._groups[0][0].value)
          info[0] = (parseInt(d3.select('#pid-input_18')._groups[0][0].value)-28)*1.0/(95-28)
          info[1] = (parseInt(d3.select('#pid-input_19')._groups[0][0].value)-1)*1.0/(130-1)
          info[2] = (parseInt(d3.select('#pid-input_20')._groups[0][0].value)-1)*1.0/(84-1)
          info[3] = (parseInt(d3.select('#pid-input_21')._groups[0][0].value)-0)*1.0/(31-0)
          var key = parseInt(d3.select('#pid-input_1').attr('key'))
          info[4+key]=1
          var key = parseInt(d3.select('#pid-input_10').attr('key'))
          info[13+key]=1
          var key = parseInt(d3.select('#pid-input_2').attr('key'))
          info[15+key]=1
          var key = parseInt(d3.select('#pid-input_11').attr('key'))
          info[19+key]=1
          var key = parseInt(d3.select('#pid-input_12').attr('key'))
          info[23+key]=1
          var key = parseInt(d3.select('#pid-input_13').attr('key'))
          info[60+key]=1
          var key = parseInt(d3.select('#pid-input_5').attr('key'))
          info[84+key]=1
          var key = parseInt(d3.select('#pid-input_6').attr('key'))
          info[95+key]=1
          var key = parseInt(d3.select('#pid-input_7').attr('key'))
          info[102+key]=1
          var key = parseInt(d3.select('#pid-input_8').attr('key'))
          info[106+key]=1
          var key = parseInt(d3.select('#pid-input_4').attr('key'))
          info[110+key]=1
          var key = parseInt(d3.select('#pid-input_14').attr('key'))
          info[112+key]=1
          var key = parseInt(d3.select('#pid-input_15').attr('key'))
          info[116+key]=1
          var key = parseInt(d3.select('#pid-input_16').attr('key'))
          info[120+key]=1
          var key = parseInt(d3.select('#pid-input_17').attr('key'))
          info[124+key]=1
          // console.log(info)
          vaxios.get('/submit',{
            params: {
              train: JSON.stringify(info)
            }
          })
            .then((res) => {
            console.log(res)
            Hub.$emit('result',res)
          })
            .catch((err) => {
              console.log(err)
            })
          
        },
        recom: function() {
          var info2 = []
          for(var i=0; i<123; i++){
            info2.push(0)
          }
          console.log(d3.select('#pid-input_18')._groups[0][0].value)
          info2[0] = (parseInt(d3.select('#pid-input_18')._groups[0][0].value)-28)*1.0/(95-28)
          info2[1] = (parseInt(d3.select('#pid-input_19')._groups[0][0].value)-1)*1.0/(130-1)
          info2[2] = (parseInt(d3.select('#pid-input_20')._groups[0][0].value)-1)*1.0/(84-1)
          info2[3] = (parseInt(d3.select('#pid-input_21')._groups[0][0].value)-0)*1.0/(31-0)
          var key = parseInt(d3.select('#pid-input_1').attr('key'))
          info2[4+key]=1
          var key = parseInt(d3.select('#pid-input_10').attr('key'))
          info2[13+key]=1
          var key = parseInt(d3.select('#pid-input_2').attr('key'))
          info2[15+key]=1
          var key = parseInt(d3.select('#pid-input_11').attr('key'))
          info2[19+key]=1
          var key = parseInt(d3.select('#pid-input_12').attr('key'))
          info2[23+key]=1
          var key = parseInt(d3.select('#pid-input_13').attr('key'))
          info2[60+key]=1
          var key = parseInt(d3.select('#pid-input_5').attr('key'))
          info2[84+key]=1
          var key = parseInt(d3.select('#pid-input_6').attr('key'))
          info2[95+key]=1
          var key = parseInt(d3.select('#pid-input_7').attr('key'))
          info2[102+key]=1
          var key = parseInt(d3.select('#pid-input_8').attr('key'))
          info2[106+key]=1
          var key = parseInt(d3.select('#pid-input_4').attr('key'))
          info2[110+key]=1
          var key = parseInt(d3.select('#pid-input_14').attr('key'))
          info2[112+key]=1
          var key = parseInt(d3.select('#pid-input_15').attr('key'))
          info2[116+key]=1
          var key = parseInt(d3.select('#pid-input_17').attr('key'))
          info2[120+key]=1
          // console.log(info)
          vaxios.get('/recom',{
            params: {
              train: JSON.stringify(info2)
            }
          })
            .then((res) => {
            console.log(res)
            Hub.$emit('re_result',res)
          })
            .catch((err) => {
              console.log(err)
            })
          
        }

      },
      created(){

      },
      mounted(){

      }
    }

</script>

