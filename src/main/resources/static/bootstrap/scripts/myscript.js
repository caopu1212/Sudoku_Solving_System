$(function () {
    //鼠标经过事件
    $('li a').hover(function () {
        //当光标进入时执行 this代表当前绑订事件的那个dom节点
        $(this).css('background-color', 'red');
    }, function () {
        //当光标移出时执行
        $(this).css('background-color', '#fff8dc');
    });


    $('#a1').toggle(function () {
        //模块1
        $(this).after("<div id='tips1'></div>");
        $('#tips1').html('<ul id="uul"><li><a  id="innera" href="uploadPage">image recognize</a></li><li><a id="innera" href="confirm2Page">enter by user</a></li><li><a id="innera" href="home1_3.html"><span class="glyphicon glyphicon-search"></span>~</a></li></ul>');
    }, function () {
        $('#tips1').remove();
    });
    //模块2
    $('#a2').toggle(function () {
        $(this).after("<div id='tips2'></div>");
        $('#tips2').html('<ul id="uul"><li><a  id="innera" href="playPage">start</a></li><li><a id="innera" href="home2_2.html">~</a></li><li><a id="innera" href="home2_3.html">~</a></li><li><a id="innera" href="home2_4.html">~</a></li></ul>');
    }, function () {
        $('#tips2').remove();
    });
    //模块3
    $('#a3').toggle(function () {
        $(this).after("<div id='tips3'></div>");
        $('#tips3').html('<ul id="uul"><li><a id="innera" href="home3_1.html">~</a></li><li><a id="innera" href="home3_2.html">~~</a></li><li><a id="innera" href="home3_3.html">~~</a></li><li><a id="innera" href="home3_4.html">~~</a></li></ul>');
    }, function () {
        $('#tips3').remove();
    });
    //模块4
    $('#a4').toggle(function () {
        $(this).after("<div id='tips4'></div>");
        $('#tips4').html('<ul id="uul"><li><a id="innera" href="ttttttt.html">~</a></li></ul>');
    }, function () {
        $('#tips4').remove();
    });
});