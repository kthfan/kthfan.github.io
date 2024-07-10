

var app = document.getElementById("app");
var iFrame = document.getElementById("iframe");
var chineseBn = document.getElementById("chinese-bn");
var englishBn = document.getElementById("english-bn");
var downloadPdfBn = document.getElementById("download-pdf-bn");
var langMode = "cn";


function refreshCN(){
    langMode = "cn";
    iFrame.height = 3300;
    
    downloadPdfBn.href = "resume-cn.pdf";

    chineseBn.classList.add("selected-lang-bn");
    englishBn.classList.remove("selected-lang-bn");
}

function refreshEN(){
    langMode = "en";
    iFrame.src = "content_en.html";
    iFrame.height = 4000;

    downloadPdfBn.href = "resume-en.pdf";

    englishBn.classList.add("selected-lang-bn");
    chineseBn.classList.remove("selected-lang-bn");
}


chineseBn.addEventListener("load", refreshCN);
englishBn.addEventListener("load", refreshEN);

chineseBn.addEventListener("click", () => {
    iFrame.src = "content_cn.html";
    refreshCN();
});

englishBn.addEventListener("click", () => {
    iFrame.src = "content_en.html";
    refreshEN();
});

iFrame.src = "content_cn.html";
refreshCN();
