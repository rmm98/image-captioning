var express = require('express');
var fileupload = require('express-fileupload');
var spawn = require('child_process').spawn
var fs = require('fs')

var app = express();

app.use(fileupload());
app.use('/assets',express.static('assets'));
app.set('view engine','ejs');

app.get('/',function(req,res){
    res.render('upload_home');
});

app.post('/upload',function(req,res){
    var image = req.files.img_upload;
    image.mv('./assets/pic.jpg',function(err){
        if (err)
            return res.status(500).send(err);
        console.log('processing');

        var pythonprocess = spawn('python',['../Final Generator/5)Generate Caption.py'],{cwd:'../Final Generator'});
        
        pythonprocess.stdout.on('data',function(data){
            res.render('caption',{data:data});
        });

        pythonprocess.stderr.on('data',function(data){
            fs.writeFile('info.txt',data,function(err){
                if(err)throw err;
            });
        });
    });
});

app.listen(3000);