var MongoClient= require("mongodb").MongoClient;
var mongodb= require('mongodb');
var uri = "mongodb+srv://clara:clara1234@cluster0-3wdze.mongodb.net/test?retryWrites=true";
var client = new MongoClient(uri, {useNewUrlParser : true});
var db;

var datalayer = {

    init : function(cb){
        client.connect(function(err){
            if(err) throw err;
            db = client.db('polytech');
            cb();
        });
    },
    
    login : function(user, cb){
        var query={$and : [{username : user.username}, {password : user.password}]};
        db.collection("User").find(query).toArray(function(err,docs){
            cb(docs);
        });
    },

    getTask : function(id, cb){
        var query={idList : id._id};
        db.collection("ToDo").find(query).toArray(function(err,docs){
            cb(docs);
        });
    },

    getList : function(id,cb){
        db.collection("Lists").find(id).toArray(function(err, docs){
            cb(docs);
        })
    },

    deleteToDo : function(id, cb){
        var query={_id : new mongodb.ObjectID(id._id)};
        db.collection("ToDo").deleteOne(query, function(err, result){
            cb();
        });
    },

    createToDo : function(task, cb){
        db.collection("ToDo").insertOne(task,function(err, result){
            cb();
        });
    },

    updateToDo : function(id, task, cb){
        var query={_id : new mongodb.ObjectID(id._id)};
        db.collection("ToDo").updateOne(query, {$set : task}, function(err, result){
            cb();
        });
    },
    createList : function(list, cb){
        db.collection("Lists").insertOne(list,function(err, result){
            cb();
        });
    },
    deleteList : function(id, cb){
        var query={_id : new mongodb.ObjectID(id._id)};
        db.collection("Lists").deleteOne(query, function(err, result){
            cb();
        });
    },
    updateList : function(id, list, cb){
        var query={_id : new mongodb.ObjectID(id._id)};
        db.collection("Lists").updateOne(query, {$set : list}, function(err, result){
            cb();
        });
    },
    verif : function(user, cb){
        var query={username : user.username};
        db.collection("User").find(query).toArray(function(err,docs){
            cb(docs);
        });
    },
    createUser : function(user, cb){
        var query = {
            username : user.username,
            password : user.password
        }
        db.collection("User").insertOne(query,function(err, result){
            cb();
        });
    },
    share : function(collab, cb){
        var query = {$addToSet: {collaborater : collab.collaborater}};
        var idList = {_id : new mongodb.ObjectID(collab.idList)};
        
        db.collection("Lists").updateOne(idList, query, function(err, result){
            cb();
        })
    }
};

module.exports=datalayer;