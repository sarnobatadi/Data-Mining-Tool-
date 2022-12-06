// Onclick of the button
// document.querySelector("button").onclick = function () {
//     // Call python's random_python function
//     eel.random_python()(function(number){					
//         // Update the div with a random number returned by python
//         document.querySelector(".random_number").innerHTML = number;
//     })
//     }
var path = ""
var setpath = ""

document.getElementById('myFile').onchange = function () {
    var name = document.getElementById('myFile'); 
    
    
    alert('Selected file: ' + name.files.item(0).name);
    path = name.files.item(0).name
    sessionStorage.setItem("path",path);
    
    console.log(setpath)
    // document.getElementById('myfile').value = setpath
  };
    
async function showCols(event,flg) {

    // // single(caliberation,output,diameter_of_indenter,applied_load,HB_value,method,lower,upper)
    setpath = sessionStorage.getItem("path");
    console.log(setpath)
    path = setpath
    var res = await eel.getCol(path)()
    console.log(path)
    console.log(res)
    var values = res;
    console.log(values)
    var select = document.createElement("select");
    select.name = "columns";
    select.id = "columns";
    select.class = "form-control";
 
    for (const val of values)
    {
        var option = document.createElement("option");
        option.value = val;
        option.text = val;
        select.appendChild(option);
    }
 
    var label = document.createElement("label");
    label.innerHTML = "Select A Column1 : "
    label.htmlFor = "columns";
 
    document.getElementById("add").appendChild(label).appendChild(select);


}

async function showCols2(event,flg) {

    // // single(caliberation,output,diameter_of_indenter,applied_load,HB_value,method,lower,upper)
    var res = await eel.getCol(path)()
    console.log(res)
    var values = res;
 
    var select = document.createElement("select");
    // select.name = "columns";
    select.id = "col2";
    select.class = "form-control";
 
    for (const val of values)
    {
        var option = document.createElement("option");
        option.value = val;
        option.text = val.charAt(0).toUpperCase() + val.slice(1);
        select.appendChild(option);
    }
 
    var label = document.createElement("label");
    label.innerHTML = "Select A Column2 : "
    label.htmlFor = "columns";
 
    document.getElementById("add2").appendChild(label).appendChild(select);


}

async function showCols3(event,flg) {

    // // single(caliberation,output,diameter_of_indenter,applied_load,HB_value,method,lower,upper)
    var res = await eel.getCol(path)()
    console.log(res)
    var values = res;
 
    var select = document.createElement("select");
    // select.name = "columns";
    select.id = "col3";
    select.class = "form-control";
 
    for (const val of values)
    {
        var option = document.createElement("option");
        option.value = val;
        option.text = val.charAt(0).toUpperCase() + val.slice(1);
        select.appendChild(option);
    }
 
    var label = document.createElement("label");
    label.innerHTML = "Select Class Column: "
    label.htmlFor = "columns";
 
    document.getElementById("add3").appendChild(label).appendChild(select);


}

async function getKNN(flg){
    
    if(flg === 0)
    {
        var colName = document.getElementById('columns').value ;
        var operationName = document.getElementById('operation').value
        var res = await eel.knn_classifier(colName,operationName)()
        document.getElementById("result").value = res
    }
    if(flg === 1)
    {
        var colName = document.getElementById('operation').value ;
        var operationName = document.getElementById('operation').value
        var res = await eel.regression_classifier(colName,operationName)()
        document.getElementById("result").value = res
    }
    if(flg === 2)
    {
        var colName = document.getElementById('operation').value ;
        var operationName = document.getElementById('operation').value
        var res = await eel.naive_baysian_classifier(colName,operationName)()
        document.getElementById("result").value = res
    }
    if(flg === 3)
    {
        var colName = document.getElementById('operation').value ;
        var operationName = document.getElementById('operation').value
        var res = await eel.annClass(operationName)()
        document.getElementById("result").value = res
    }
    if(flg === 4)
    {
        var colName = document.getElementById('columns').value ;
        var operationName = document.getElementById('operation').value
        var res = await eel.kmncluster(colName,operationName)()
        document.getElementById("result").value = res
    }
    if(flg === 5)
    {
        var colName = document.getElementById('columns').value ;
        var operationName = document.getElementById('operation').value
        var res = await eel.kmdcluster(colName,operationName)()
        document.getElementById("result").value = res
    }
    if(flg === 6)
    {
        var colName = document.getElementById('columns').value ;
        var operationName = document.getElementById('operation').value
        var res = await eel.hicluster(colName,operationName)()
        document.getElementById("result").value = res
    }
    if(flg === 7)
    {
        var colName = document.getElementById('columns').value ;
        var operationName = document.getElementById('operation').value
        var res = await eel.aproiri(colName,operationName)()
        document.getElementById("result").value = res
    }
    if(flg === 8)
    {
        var colName = document.getElementById('columns').value ;
        var operationName = document.getElementById('operation').value
        var res = await eel.pgrank(colName,operationName)()
        document.getElementById("result").value = res
    }
    if(flg === 9)
    {
        var colName = document.getElementById('columns').value ;
        // var operationName = document.getElementById('operation').value
        var res = await eel.hits(colName)()
        document.getElementById("result").value = res
    }
    if(flg === 10)
    {
        var colName = document.getElementById('columns').value ;
        var operationName = document.getElementById('operation').value
        var res = await eel.crawl(colName,operationName)()
        document.getElementById("result").value = res
    }
    if(flg === 11)
    {
        var colName = document.getElementById('columns').value ;
        var operationName = document.getElementById('operation').value
        var res = await eel.dbres(colName,operationName)()
        document.getElementById("result").value = res
    }
    
    

}


async function getResult(){
    var colName = document.getElementById('columns').value ;
    var operationName = document.getElementById('operation').value

    var res = await eel.getRes(colName,operationName)()

    document.getElementById("result").value = res

}

async function getResult2(){
    var col1 = document.getElementById('columns').value ;
    var operationName = document.getElementById('operation').value
    var col2 = document.getElementById('col2').value
    var col3 = document.getElementById('col3').value

    var res = await eel.getRes2(col1,col2,col3,operationName)()

    document.getElementById("result").value = res

}


async function viewData(){
    
    var res = await eel.showData()()    

}