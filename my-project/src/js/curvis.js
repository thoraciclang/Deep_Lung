export const Curvis = function() {
    var curvis = {},
        container = null,
        data = {},
        size = [400, 250],
        dispatch = d3.dispatch('save','change','start')

    curvis.container = function (_) {
        if (!arguments.length) return container;
        container = _;
        return curvis;
    }

    curvis.data = function(_) {
        if (!arguments.length) return data;
        data = _;
        return curvis;
    }

    curvis.size = function(_) {
        if (!arguments.length) return size;
        size = _;
        return curvis;
    }

    curvis.dispatch = dispatch;

    let canvas
    let graph = []
    let line
    let x_scale
    let y_scale
    let x_time
    let xAxis
    let yAxis

    curvis.init = function() {
        container
            .attr('width', size[0])
            .attr('height', size[1])
        container.selectAll().remove();

        canvas = container.append('g')
          .attr('class', 'canvas')
          .attr('transform', `translate(60, 10)`)
        x_scale = d3.scaleLinear().range([0, 300]).domain([0,60])
        x_time = d3.scaleLinear().range([0, 300]).domain([0,5])
        y_scale = d3.scaleLinear().range([220, 0]).domain([0,1])
        xAxis = d3.axisBottom(x_time)
        yAxis = d3.axisLeft(y_scale)
        canvas.append("g")
        .attr("class", "axis axis--x")
        .attr("transform", "translate(0,220)")
        .call(xAxis);

        canvas.append("g")
        .attr("class", "axis axis--y")
        .call(yAxis);
        return curvis
    }

    curvis.layout = function() {
        graph = []
        for(var i=0; i<data[0].length-12; i++){
            graph.push({x: data[0][i], y:data[1][i]})
        }
        line = d3.line()
            .x((d)=> x_scale(d.x))
            .y((d)=> y_scale(d.y))
            
        return curvis
    }

    curvis.update = function() {
        
        canvas
            .append("path")
            .attr("fill-opacity",0)
            .attr('stroke','orange')
            .attr('stroke-width', 2)
            .attr('d', line(graph))
  
        return curvis
    }

    return curvis
}
