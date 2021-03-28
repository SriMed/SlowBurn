var burn = require("./burn");

exports.doSet = function(app) {
  //-------express 'get' handlers
  app.get("/", burn.burn_home);
  app.get("/predict", burn.render_prediction);
};

