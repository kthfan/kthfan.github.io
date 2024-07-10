

var app = document.getElementById("app");
var iFrame = document.getElementById("iframe");
var chineseBn = document.getElementById("chinese-bn");
var englishBn = document.getElementById("english-bn");
var downloadPdfBn = document.getElementById("download-pdf-bn");
var langMode = "zh-tw";


function refreshCN(){
    langMode = "zh-tw";
    iFrame.height = 3300;
    
    downloadPdfBn.href = "resume-zh-tw.pdf";

    chineseBn.classList.add("selected-lang-bn");
    englishBn.classList.remove("selected-lang-bn");
}

function refreshEN(){
    langMode = "en-us";
    iFrame.src = "content-en-us.html";
    iFrame.height = 4000;

    downloadPdfBn.href = "resume-en-us.pdf";

    englishBn.classList.add("selected-lang-bn");
    chineseBn.classList.remove("selected-lang-bn");
}


chineseBn.addEventListener("load", refreshCN);
englishBn.addEventListener("load", refreshEN);

chineseBn.addEventListener("click", () => {
    iFrame.src = "content-zh-tw.html";
    refreshCN();
});

englishBn.addEventListener("click", () => {
    iFrame.src = "content-en-us.html";
    refreshEN();
});

iFrame.src = "content-zh-tw.html";
refreshCN();
