
//Currently not using this file
var canvas = document.querySelector('canvas');

document.write(canvas)
//canvas.width  = window.innerWidth;
//canvas.height = window.innerHeight;

var c = canvas.getContext('2d');
/*
c.fillStyle = 'rgba(255,0,0,0.1)';
c.fillRect(100,100,100,100)
c.fillStyle = 'rgba(0,255,128,0.1)';
c.fillRect(200,200,100,100)
c.fillStyle = 'rgba(25,128,255,0.1)';
c.fillRect(300,300,100,100)
console.log(canvas);
*/

/*
//Line
c.beginPath();
c.moveTo(50,300);
c.lineTo(300,100);
c.lineTo(400,300)
c.strokeStyle = "#fa3424"
c.stroke();
*/

/*
//Arc /Circle
for (var i=0;i<100;i++){
    var x = Math.random() * window.innerWidth
    var y = Math.random() * window.innerHeight 
    c.beginPath();
    c.arc(x,y,30,0, Math.PI*2,false);
    var color =  Math.random() * 255;
    c.strokeStyle = 'rgba(0,color,color,0.5)'
    c.stroke();
}
*/

var mouse = {
    x: undefined,
    y: undefined
}

var maxRadius = 40;
var minRadius = 5;

var colorArray = [
    '#1E4363',
    '#FCF2CB',
    '#FFB00D',
    '#FF8926',
    '#BC2D19'
];

window.addEventListener('mousemove',
    function(event){
        mouse.x = event.x;
        mouse.y = event.y;
})

window.addEventListener('resize',
    function(){
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        init();
    })
/*var x = Math.random()*innerWidth;
var y = Math.random()*innerHeight;
var dx = (Math.random() - 0.5)*5;
var dy = (Math.random() - 0.5)*5;
var radius = 30;*/

function Circle(x,y,dx,dy,radius){
    this.x = x;
    this.y = y;
    this.dx = dx;
    this.dy = dy;
    this.radius = radius;
    this.minRadius = radius;
    this.color = colorArray[Math.floor(Math.random()* colorArray.length)]

    this.draw = function(){
        c.beginPath();
        c.arc(this.x,this.y,this.radius,0, Math.PI*2,false);
        c.fillStyle = this.color
        c.stroke();
        c.fill();
        
        //console.log('circle draw')
    }

    this.update = function()
    {
        if(this.x + radius> innerWidth || this.x-radius < 0)
        {
            this.dx = -this.dx;
        }
        if(this.y+radius>innerHeight || this.y-radius<0)
        {
            this.dy = -this.dy;
        }
        this.x+= this.dx;
        this.y+= this.dy;

        // interactivity
        if(mouse.x - this.x < 50 && mouse.x - this.x>-50
            && mouse.y-this.y<50 && mouse.y-this.y >-50)
        {
            if(this.radius < maxRadius)
            {
                this.radius +=1;
            }
        }
        else if (this.radius> this.minRadius)
        {
            this.radius -=1;
        }

        this.draw();
    }

}



var circleArray = [];
function init(){
    circleArray = [];
    for (var i=0;i<500;i++){
        var radius = Math.random() * 3 + 1;
        var x = Math.random()*(innerWidth - radius*2) + radius;
        var y = Math.random()*(innerHeight- radius*2) + radius;
        var dx = (Math.random() - 0.5)*3;
        var dy = (Math.random() - 0.5)*3;
    
        circleArray.push(new Circle(x,y,dx,dy,radius));
        }
}

init()

function animate(){
    requestAnimationFrame(animate);
    c.clearRect(0,0,innerWidth,innerHeight);
    for (var i=0; i <circleArray.length;i++)
    {
        circleArray[i].update();
    }
}

animate();
