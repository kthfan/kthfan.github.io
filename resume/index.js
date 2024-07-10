

var app = document.getElementById("app");
var iFrame = document.getElementById("iframe");
var chineseBn = document.getElementById("chinese-bn");
var englishBn = document.getElementById("english-bn");
var downloadPdfBn = document.getElementById("download-pdf-bn");
var qrCodeImg = document.getElementById("qr-code");
var langMode = "zh-tw";
var qrCodeEnlarged = false;


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

function togggleQRCode(){
    if(qrCodeEnlarged){
        qrCodeImg.width = 50;
        qrCodeEnlarged = false;
    } else{
        qrCodeImg.width = 300;
        qrCodeEnlarged = true;
    }
}

qrCodeImg.addEventListener("click", () => {
    if(qrCodeEnlarged){
        qrCodeImg.width = 50;
        qrCodeEnlarged = false;
    } else{
        qrCodeImg.width = 300;
        qrCodeEnlarged = true;
    }
});
qrCodeImg.addEventListener("mouseover", () => {
    if(qrCodeEnlarged){
        qrCodeImg.width = 200;
    } else{
        qrCodeImg.width = 100;
    }
});
qrCodeImg.addEventListener("mouseleave", () => {
    if(qrCodeEnlarged){
        qrCodeImg.width = 300;
    } else{
        qrCodeImg.width = 50;
    }
});



iFrame.src = "content-zh-tw.html";
refreshCN();
