var demoApp = angular.module('TodoApp',['ngCookies']); //[cookies]
var serv = "https://todolist-bellin.herokuapp.com";

demoApp.controller('MainController', function ($scope, $http, $cookies){ //,$cookies
    $scope.updateData = {};
    $scope.updateListe = {};
    $scope.listeData = {};
    $scope.listeDataP = {};
    $scope.tache = [];
    $scope.tacheP = [];
    $scope.nameTask={};
    $scope.user = {};
    $scope.collab = {};

    
    getTask = function(id, idx){        
        $scope.listeData.idListe = id;      
		$http.post('/getTask', $scope.listeData)
			.success(function(data) {
                $scope.tache[idx] = data;
			})
			.error(function(data) {
				console.log('Error: ' + data);
            });
    }

    getTaskP = function(id, idx){        
        $scope.listeDataP.idListe = id;      
		$http.post('/getTask', $scope.listeDataP)
			.success(function(data) {
                $scope.tacheP[idx] = data;
			})
			.error(function(data) {
				console.log('Error: ' + data);
            });
    }

    getListe = function(){
        $scope.user =$cookies;
        var req = {author : $cookies.id};
        $http.post('/getList',req).then(function(resp){
            console.log(" 1 : ",resp);
            $scope.Liste = resp.data;
            i=0;
            for(l of resp.data){
                getTask(l._id, i);
                i++;
            }
        });
        var req = {collaborater : $cookies.id};
        $http.post('/getListP',req).then(function(resp){
            console.log(" 2 : ", resp);
            $scope.ListeP = resp.data;
            i=0;
            for(l of resp.data){
                getTaskP(l._id, i);
                i++;
            }
        });    
    }

    //lancée au chargement de la page: check des cookies, récupération des listes
    if($cookies.user!=null){
        $scope.user =$cookies;
        var req = {author : $cookies.id};
        $http.post('/getList',req).then(function(resp){
            console.log(" 1 : ",resp);
            $scope.Liste = resp.data;
            i=0;
            for(l of resp.data){
                getTask(l._id, i);
                i++;
            }
        });
        var req = {collaborater : $cookies.id};
        $http.post('/getListP',req).then(function(resp){
            console.log(" 2 : ", resp);
            $scope.ListeP = resp.data;
            i=0;
            for(l of resp.data){
                getTaskP(l._id, i);
                i++;
            }
        });
    }else{
        document.location.href= serv + "/connection.html";
    }    
 
    $scope.createToDo = function(id){
        var req = {
            name : $scope.nameTask.text,
            idList : id
        };
		$http.post('/createToDo',req)
		.success(function(data) {
            $scope.nameTask={};
           getListe();
		})
		.error(function(data) {
			console.log('Error: ' + data);
        });   
    }

    $scope.deleteToDo = function(id){     
        var req = {
            identifiant : id 
        };
		$http.post('/deleteToDo', req)
			.success(function(data) {
				getListe();
			})
			.error(function(data) {
				console.log('Error: ' + data);
			});
    }

    $scope.checkToDo =function(id, ok){
        $scope.updateData.identifiant = id;
        $scope.updateData.ok = !ok;
        $http.post('/checkToDo', $scope.updateData).success(function(data) {    
            getListe();
        })
        .error(function(data) {
            console.log('Error: ' + data);
        });
    }

    $scope.updateToDo = function(){
        $scope.updateData.identifiant = $scope.toggleUT._id;
        console.log($scope.updateData.text);
        if($scope.updateData.text!=undefined){
            $http.post('/updateToDo', $scope.updateData).success(function(data) {    
                getListe();
                $scope.updateData = {};
                $scope.toggleUT = {};
            })
            .error(function(data) {
                console.log('Error: ' + data);
            });
        }
    }

    $scope.createList = function(){
        if($scope.nameList!= undefined){
            var req = {
                nameList : $scope.nameList,
                author : $cookies.id
            };

            $http.post('/createList',req)
            .success(function(data) {
                $scope.nameList = "";
                getListe();
            })
            .error(function(data) {
                console.log('Error: ' + data);
            }); 
        }
    }

    $scope.deleteList = function(id){     
        var req = {
            identifiant : id 
        };
		$http.post('/deleteList', req)
			.success(function(data) {
				getListe();
			})
			.error(function(data) {
				console.log('Error: ' + data);
			});
    }

    $scope.updateList = function(){
        $scope.updateListe.identifiant= $scope.toggleU._id;
        if($scope.updateListe.text!=undefined){
            $http.post('/updateList', $scope.updateListe).success(function(data) { 
                $scope.updateListe = {}; 
                getListe();
                $scope.toggleU = {};
            })
            .error(function(data) {
                console.log('Error: ' + data);
            });
        }
    }

    //partage d'une liste avec des collaborateurs
    $scope.share = function(){
        var req = {username : $scope.collab.text};
        $http.post('/verif',req).then(function(resp){
            if(resp.data.length>0){
                req = {
                    _id : $scope.toggleS._id,
                    collaborater : resp.data[0]._id
                }
                $http.post('/share', req).success(function(data){
                    window.alert("liste : "+$scope.toggleS.name+" partagée avec "+ resp.data[0].username);
                    $scope.collab.text ="";
                    $scope.toggleS = {};
                    getListe();
                })
            }else{
                window.alert("ce utilisateur n'existe pas");
            }
        });
    }

    $scope.deconnexion = function(){
        delete $cookies['user'];
        document.location.href= serv + "/connection.html";
    }

    $scope.shareToggle = function(list){
        $scope.toggleS = list;
    }

    $scope.updateToggle = function(list){

        $scope.toggleU = list;
    }

    $scope.updateTaskToggle = function(task){
        console.log(task);
        $scope.toggleUT = task;
    }

    $scope.updateTaskPToggle = function(task){
        console.log(task);
        $scope.toggleUTP = task;
    }

});

var connectionApp = angular.module('connection', ['ngCookies']);
connectionApp.controller('ConnectionController', function ($scope, $http, $cookies){
   $scope.password = "";
   $scope.username = "";
     if($cookies.user!=null){
        document.location.href= serv;
     }
    $scope.auth = function(){
        if($scope.username && $scope.password){
            var req = {
                username : $scope.username,
                password : $scope.password
            };

            $http.post('/login', req).then(function(resp){
                if(resp.data.length>0){
                    $cookies.user=resp.data[0].username;
                    $cookies.id = resp.data[0]._id;
                    document.location.href= serv;
                }
                else{
                    window.alert("Coordonnées incorrectes");
                }
                $scope.password="";
                $scope.username="";
            });
        }
    };

    $scope.inscription = function(){
        if($scope.usernameCreate && $scope.passwordCreate){
            var req = {
                username : $scope.usernameCreate,
                password : $scope.passwordCreate
            };

            $http.post('/verif',req).then(function(resp){
                if(resp.data.length==0){
                    $http.post('/createUser',req).then(function(resp){

                        $scope.usernameCreate = "";
                        $scope.passwordCreate = "";
                        window.alert("Vous être inscrit, vous pouvez maintenant vous connecter");

                    });
                }else{
                    window.alert("Ce nom est déjà prit");
                }
            });
        }
    };

});
