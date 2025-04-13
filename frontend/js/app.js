// frontend/js/app.js
const generateButton = document.getElementById("generateButton");
const regenerateButton = document.getElementById("regenerateButton");
const loadingMessage = document.getElementById("loadingMessage");
const resultContainer = document.getElementById("resultContainer");

const seedInput = document.getElementById("seedInput");
const characterTypeSelect = document.getElementById("characterType");
const colorThemeSelect = document.getElementById("colorTheme");

function sendGenerationRequest() {
  loadingMessage.style.display = "block";
  resultContainer.innerHTML = "";
  regenerateButton.style.display = "none";
  
  const seed = seedInput.value;
  const characterType = characterTypeSelect.value;
  const colorTheme = colorThemeSelect.value;
  
  const params = {
    seed: seed || null,
    character_type: characterType,
    color_theme: colorTheme
  };
  
  console.log("发送请求，参数：", params);
  
  fetch("http://127.0.0.1:5000/api/generate-character", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params)
  })
  .then(response => {
    console.log("响应状态：", response.status);
    return response.json();
  })
  .then(data => {
    console.log("返回数据：", data);
    loadingMessage.style.display = "none";
    if (data.image_url) {
      const img = document.createElement("img");
      img.src = data.image_url;
      img.alt = "生成的角色图片";
      resultContainer.innerHTML = "";
      resultContainer.appendChild(img);
      regenerateButton.style.display = "inline-block";
    } else {
      resultContainer.innerHTML = "生成失败，请重试！";
    }
  })
  .catch(err => {
    loadingMessage.style.display = "none";
    console.error("请求错误：", err);
    alert("请求出错：" + err);
  });
}

generateButton.addEventListener("click", sendGenerationRequest);
regenerateButton.addEventListener("click", sendGenerationRequest);

document.getElementById("submitFeedback").addEventListener("click", function() {
  const feedback = document.getElementById("feedbackText").value;
  console.log("用户反馈：", feedback);
  alert("谢谢您的反馈！");
});
