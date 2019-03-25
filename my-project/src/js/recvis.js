export const Recvis = function() {
    var recvis = {},
        container = null,
        data = {},
        size = [400, 250],
        dispatch = d3.dispatch('save','change','start')

    recvis.container = function (_) {
        if (!arguments.length) return container;
        container = _;
        return recvis;
    }

    recvis.data = function(_) {
        if (!arguments.length) return data;
        data = _;
        return recvis;
    }

    recvis.size = function(_) {
        if (!arguments.length) return size;
        size = _;
        return recvis;
    }

    recvis.dispatch = dispatch;

    let canvas
    let graph = [[],[]]
    let line
    let x_scale
    let y_scale
    let x_time
    let xAxis
    let yAxis

    recvis.init = function() {
        container
            .attr('width', size[0])
            .attr('height', size[1])
        container.selectAll().remove();

        canvas = container.append('g')
          .attr('class', 'canvas')
          .attr('transform', `translate(60, 10)`)
        x_scale = d3.scaleLinear().range([0, 300]).domain([0,36])
        x_time = d3.scaleLinear().range([0, 300]).domain([0,3])
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
        return recvis
    }

    recvis.layout = function() {
        graph = [[], []]
        console.log(data)
        for(var i=0; i<data.length; i++){
            for(var j=0; j<36; j++){
                graph[i].push({x: data[i][0][j], y:data[i][1][j]})
            }
            
        }
        line = d3.line()
                .x((d)=> x_scale(d.x))
                .y((d)=> y_scale(d.y))
        
        return recvis
    }

    recvis.update = function() {
        
        canvas
            .append("path")
            .attr("fill-opacity",0)
            .attr('stroke','blue')
            .attr('stroke-width', 2)
            .attr('d', line(graph[0]))
        
        canvas
            .append("path")
            .attr("fill-opacity",0)
            .attr('stroke','red')
            .attr('stroke-width', 2)
            .attr('d', line(graph[1]))

        legend = canvas.append("g")
            .attr("class","legend")
            .attr("transform","translate(50,30)")
            .style("font-size","12px")
        
        legend.append('circle')
            .attr('r',2)
            .attr('fill', 'blue')
        legend.append('text')
            .text('Lobectomy')
        legend.append('circle')
            .attr('r',2)
            .attr('fill', 'red')
        legend.append('text')
            .text('Sublobar')
  
        return recvis
    }

    return recvis
}
