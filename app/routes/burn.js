var path = require("path");
var child_process = require("child_process");

// function anagramCheck(req, res, next) {
//   var word = req.query.word;
//   word = word.toLowerCase();
//   res.locals.word = word;

//   var pa = __dirname;
//   pa = pa.slice(0, pa.lastIndexOf("/"));

//   var spawn = child_process.spawn;

//   var py_script = path.join(pa, "resources/get_prediction.py");
//   console.log("python path at line 119");
//   console.log(py_script);

//   var process = spawn("python", [py_script, word, letters]);

//   var result = "";

//   process.stdout.on("data", function(data) {
//     result = data.toString();
//     console.log("anagramCheck result at line 130 " + result);

//     res.locals.ac = result;
//     console.log("res.locals arg");
//     console.log(res.locals.ac);
//     next();
//   });
// }

exports.burn_home = function(req, res) {
  var render_dict = {
    message: "Welcome to Steady Burn!"
  };
  res.render("burn_home", render_dict);
};

function predict_burn(req, res, next) {
  var des = req.query.des; //this is how you get input from ajax
  var resource = req.query.resource;
  var men = req.query.men;
  var spawn = child_process.spawn;

  console.log("calling python script");
  var process = spawn("python", ["resources/get_prediction.py", des, resource, men]); 
  var result = "";
  
  console.log("python script spawned")
  
  process.stdout.on("data", function(data) {
    result = data.toString();
    console.log("Python script returned:" + result);

    res.locals.ac = result;
    console.log("res.locals arg");
    console.log(res.locals.ac);
    next();
  });
}

exports.render_prediction = [predict_burn, function(req, res) {
    console.log("in render_prediction");
    var render_dict = {
      message: "Welcome to Steady Burn!",
      prediction: res.locals.ac
    };
    res.render("prediction", render_dict); // #populate render_dict
  }
];