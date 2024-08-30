onUiLoaded(setupStylez);
let orgPrompt = '';
let orgNegative = '';
let tabname = '';
let promptNeg = '';
let promptPos = '';
let arwidth = '';
let arheight = '';

function setupStylez() {
    //create new button (t2i)
    const t2i_StyleBtn = document.createElement("button");
    t2i_StyleBtn.setAttribute("class", "lg secondary gradio-button tool svelte-cmf5ev");
    t2i_StyleBtn.setAttribute("id", "t2i_stylez_btn");
    t2i_StyleBtn.setAttribute("onClick", "showHideStylez()");
    t2i_StyleBtn.innerText = `🎨`;
    //add new button
    const txt2img_tools = gradioApp().getElementById("txt2img_clear_prompt");
    txt2img_tools.parentNode.appendChild(t2i_StyleBtn);
    //create new button (i2i)
    const i2i_StyleBtn = document.createElement("button");
    i2i_StyleBtn.setAttribute("class", "lg secondary gradio-button tool svelte-cmf5ev");
    i2i_StyleBtn.setAttribute("id", "i2i_stylez_btn");
    i2i_StyleBtn.setAttribute("onClick", "showHideStylez()");
    i2i_StyleBtn.innerText = `🎨`;
    //add new button
    const img2img_tools = gradioApp().getElementById("img2img_clear_prompt");
    img2img_tools.parentNode.appendChild(i2i_StyleBtn);
    //Setup Browser
    const hideoldbar = gradioApp().querySelector('#hide_default_styles > label > input');
    if (hideoldbar.checked === true) {
        hideOldStyles(true);
    } else {
        hideOldStyles(false);
    }
    
    const t2i_stylez_container = gradioApp().querySelector("#Stylez");
    console.log(t2i_stylez_container)
    const tabs = gradioApp().getElementById("tabs");
    tabs.appendChild(t2i_stylez_container);
    const tabNav = document.querySelector(".tab-nav");
    if (tabNav) {
      const buttonTextToFind = "stylez_menutab";
      const buttons = tabNav.querySelectorAll("button.svelte-kqij2n");
      let styleztabbtn = null;
      buttons.forEach(button => {
        if (button.innerText.trim() === buttonTextToFind) {
            styleztabbtn = button;
        }
      });
      if (styleztabbtn) {
        styleztabbtn.style.display = "none";
      } 
    }
}

function hideOldStyles(bool) {
    if(bool == true){
        const stylesOld_t2i = gradioApp().getElementById("txt2img_styles_row");
        stylesOld_t2i.style.display = 'none';
        const stylesOld_i2i = gradioApp().getElementById("img2img_styles_row");
        stylesOld_i2i.style.display = 'none';
    } else {
        const stylesOld_t2i = gradioApp().getElementById("txt2img_styles_row");
        stylesOld_t2i.style.display = 'block';
        const stylesOld_i2i = gradioApp().getElementById("img2img_styles_row");
        stylesOld_i2i.style.display = 'block';
    }
}

function showHideStylez() {
    const stylez = gradioApp().getElementById("Stylez");
    const computedStyle = window.getComputedStyle(stylez);
    if (computedStyle.getPropertyValue("display") === "none" || computedStyle.getPropertyValue("visibility") === "hidden") {
        stylez.style.display = "block";
    } else {
        stylez.style.display = "none";
    }
}
// get active tab
function getENActiveTab() {
    let activetab = "";
    const tab = gradioApp().getElementById("tab_txt2img");
    const computedStyle = window.getComputedStyle(tab);
    if (computedStyle.getPropertyValue("display") === "none" || computedStyle.getPropertyValue("visibility") === "hidden") {
        activetab = "img2img";
    } else {
        activetab = "txt2img";
    }
    return (activetab);
}

function tabCheck(mutationsList, observer) {
    for (const mutation of mutationsList) {
        if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
            const tabTxt2img = gradioApp().getElementById('tab_txt2img');
            const tabImg2img = gradioApp().getElementById('tab_img2img');
            const stylez = gradioApp().getElementById("Stylez");
            // Check if the display property of tab_txt2img is 'none'
            if (tabTxt2img.style.display === 'none') {
                stylez.style.display = "none";
                tabname = getENActiveTab();
                promptPos = gradioApp().querySelector(`#${tabname}_prompt > label > textarea`);
                promptNeg = gradioApp().querySelector(`#${tabname}_neg_prompt > label > textarea`);
            }
            // Check if the display property of tab_img2img is 'none'
            if (tabImg2img.style.display === 'none') {
                stylez.style.display = "none";
                tabname = getENActiveTab();
                promptPos = gradioApp().querySelector(`#${tabname}_prompt > label > textarea`);
                promptNeg = gradioApp().querySelector(`#${tabname}_neg_prompt > label > textarea`);
            }
        }
    }
}
//check to see if gradio is fully loaded
function checkElement() {
    const tabTxt2img = gradioApp().getElementById('tab_txt2img');
    if (tabTxt2img) {
        const observer = new MutationObserver(tabCheck);
        tabname = getENActiveTab();
        promptPos = gradioApp().querySelector(`#${tabname}_prompt > label > textarea`);
        promptNeg = gradioApp().querySelector(`#${tabname}_neg_prompt > label > textarea`);
        const tab_txt2img = gradioApp().getElementById('tab_txt2img');
        const tab_img2img = gradioApp().getElementById('tab_img2img');
        const config = {attributes: true};
        observer.observe(tab_txt2img, config);
        observer.observe(tab_img2img, config);
        const style_savefolder_temp = gradioApp().querySelector("#style_savefolder_temp > label > textarea");
        applyValues(style_savefolder_temp,"Styles")
        gradioApp().getElementById('style_save_btn').addEventListener('click', () => {
            saveRefresh();
        });
        gradioApp().getElementById('style_delete_btn').addEventListener('click', () => {
            deleteRefresh();
        });
    } else {
        setTimeout(checkElement, 100);
    }
}
checkElement();

// 添加选中框
function toggleCardSelection(event, folder, filename) {
    const card = event.target.closest('.style_card');
    const checkbox = card.querySelector('.style_card_checkbox');
    checkbox.classList.toggle('checked');
    card.classList.toggle('selected');
    // 阻止传播以防点击事件传播到卡片本身
    event.stopPropagation();
}

//apply styles
function applyStyle(prompt, negative,origin) {
    const applyStylePrompt = gradioApp().querySelector('#styles_apply_prompt > label > input');
    const applyStyleNeg = gradioApp().querySelector('#styles_apply_neg > label > input');
    //positive checks
    orgPrompt = promptPos.value;
    orgNegative = promptNeg.value;
    if (origin == "Stylez") {
        prompt = removeFirstAndLastCharacter(prompt)
        negative = removeFirstAndLastCharacter(negative)
        if(prompt.includes("{prompt}")) {
            const promptPossections = prompt.split("{prompt}");
            const promptPossectionA = promptPossections[0].trim();
            const promptPossectionB = promptPossections[1].trim();
            if (orgPrompt.includes(promptPossectionA) & orgPrompt.includes(promptPossectionB)) {
                orgPrompt = orgPrompt.replace(promptPossectionA,"");
                orgPrompt = orgPrompt.replace(promptPossectionB,"");
                orgPrompt = orgPrompt.replace(/^\s+/, "");
                orgPrompt = orgPrompt.replace(/^,+/g, "");
                orgPrompt = orgPrompt.replace(/^\s+/, "");
                if (applyStylePrompt.checked === true)
                {
                    applyValues(promptPos,orgPrompt)
                }
            } else {
                appendStyle(applyStylePrompt,prompt,orgPrompt,promptPos)
            }
        } else {
                if (prompt !== "") {
                    if (orgPrompt.includes(prompt) || orgPrompt.includes(", "+ prompt)) {
                        if(orgPrompt.includes(prompt)) {}
                        orgPrompt = orgPrompt.replace(", "+ prompt,"");
                        orgPrompt = orgPrompt.replace(prompt,"");
                        orgPrompt = orgPrompt.replace(/^\s+/, "");
                        orgPrompt = orgPrompt.replace(/^,+/g, "");
                        orgPrompt = orgPrompt.replace(/^\s+/, "");
                        if (applyStylePrompt.checked === true)
                        {
                            applyValues(promptPos,orgPrompt)
                        }
                    } else {
                        appendStyle(applyStylePrompt,prompt,orgPrompt,promptPos)
                    }
                }
            }
            if (negative !== "") {
                if (orgNegative.includes(negative) || orgNegative.includes(", "+ negative)) {
                    if(orgNegative.includes(negative)) {}
                    orgNegative = orgNegative.replace(", "+ negative,"");
                    orgNegative = orgNegative.replace(negative,"");
                    orgNegative = orgNegative.replace(/^\s+/, "");
                    orgNegative = orgNegative.replace(/^,+/g, "");
                    orgNegative = orgNegative.replace(/^\s+/, "");
                    if (applyStyleNeg.checked === true)
                    {
                        applyValues(promptNeg,orgNegative)
                    }
            
                } else {
                    appendStyle(applyStyleNeg,negative,orgNegative,promptNeg)
                }
            }
    } else {
        prompt = decodeURIComponent(prompt).replaceAll(/%27/g, "'")
        negative = decodeURIComponent(negative).replaceAll(/%27/g, "'")
        if (orgPrompt.includes(prompt) || orgPrompt.includes(", "+ prompt)) {
            if(orgPrompt.includes(prompt)) {}
            orgPrompt = ""
            if (applyStylePrompt.checked === true)
            {
                applyValues(promptPos,orgPrompt)
            }
        } else {
            appendStyle(applyStylePrompt,prompt,"",promptPos)
        }
        if (orgNegative.includes(negative) || orgNegative.includes(", "+ negative)) {
            if(orgNegative.includes(negative)) {}
            orgNegative = ""
            if (applyStyleNeg.checked === true)
            {
                applyValues(promptNeg,orgNegative)
            }
        } else {
            appendStyle(applyStyleNeg,negative,"",promptNeg)
        }
    }
    
    // 应用选中框
    const card = event.target.closest('.style_card');
    // 点击卡片时切换选择
    toggleCardSelection(event, card.getAttribute('data-foldername'), card.getAttribute('data-filename'));
}

function hoverPreviewStyle(prompt,negative,origin) {
    const enablePreviewChk = gradioApp().querySelector('#HoverOverStyle_preview > label > input');
    const enablePreview = enablePreviewChk.checked;
    if (enablePreview === true) { 
        previewbox = gradioApp().getElementById("stylezPreviewBoxid");
        previewbox.style.display = "block";
        if (origin == "Stylez") {
            prompt = removeFirstAndLastCharacter(prompt)
            negative = removeFirstAndLastCharacter(negative)
        } else {
            prompt = decodeURIComponent(prompt).replaceAll(/%27/g, "'")
            negative = decodeURIComponent(negative).replaceAll(/%27/g, "'")
        }
        if (prompt == ""){
            prompt = "NULL"
        }
        if (negative == ""){
            negative = "NULL"
        }
        pos = gradioApp().getElementById("stylezPreviewPositive");
        neg = gradioApp().getElementById("stylezPreviewNegative");
        pos.textContent = "Prompt: " + prompt;
        neg.textContent = "Negative: " + negative;
    }
}

function hoverPreviewStyleOut() {
    previewbox = gradioApp().getElementById("stylezPreviewBoxid");
    pos = gradioApp().getElementById("stylezPreviewPositive");
    neg = gradioApp().getElementById("stylezPreviewNegative");
    pos.textContent = "Prompt: ";
    neg.textContent = "Negative: ";
    previewbox.style.display = "none";
}
function appendStyle(applyStyle,prompt,oldprompt,promptbox) {
if (applyStyle.checked === true) {
        if (prompt.includes("{prompt}")) {
            oldprompt = promptbox.value;
            prompt = prompt.replace('{prompt}', oldprompt);
            promptbox.value = prompt;
            updateInput(promptbox);
        } else {
            if (oldprompt === '') {
                oldprompt = prompt;
                promptbox.value = prompt;
                updateInput(promptbox);
            } else {
                promptbox.value = oldprompt + ", " + prompt;
            }
            updateInput(promptbox);
        }
    }
}

function applyValues(a, b) {
    a.value = b;
    updateInput(a);
}

function removeFirstAndLastCharacter(inputString) {
    if (inputString.length >= 2) {
        return inputString.slice(1, -1);
    } else {
        inputString = "";
        return inputString;
    }
}

function cardSizeChange(value) {
    const styleCards = gradioApp().querySelectorAll('.style_card');
    styleCards.forEach((card) => {
        card.style.minHeight = value + 'px';
        card.style.maxHeight = value + 'px';
        card.style.minWidth = value + 'px';
        card.style.maxWidth = value + 'px';
    });
}

function filterSearch(cat, search) {
    let searchString = search.toLowerCase();
    const styleCards = gradioApp().querySelectorAll('.style_card');
    if (searchString == "") {
        if (cat == "All") {
            styleCards.forEach(card => {
                card.style.display = "flex";
            });
        } else if (cat == "Favourites") {
            styleCards.forEach(card => {
                var btn = card.querySelector(".favouriteStyleBtn");
                let computedelem = getComputedStyle(btn);
                if (computedelem.color === "rgb(255, 255, 255)") {
                    card.style.display = "none";
                } else {
                    card.style.display = "flex";
                }
            });
        } else {
            styleCards.forEach(card => {
                const cardCategory = card.getAttribute('data-category');
                if (cardCategory == cat) {
                    card.style.display = "flex";
                } else {
                    card.style.display = "none";
                }
            });
        }
    } else {
        if (cat == "All") {
            styleCards.forEach(card => {
                const cardTitle = card.getAttribute('data-title');
                if (cardTitle.includes(searchString)) {
                    card.style.display = "flex";
                } else {
                    card.style.display = "none";
                }
            });
        } else if (cat == "Favourites") {
            styleCards.forEach(card => {
                const cardTitle = card.getAttribute('data-title');
                var btn = card.querySelector(".favouriteStyleBtn");
                let computedelem = getComputedStyle(btn);
                if (cardTitle.includes(searchString)) {
                    if (computedelem.color === "rgb(255, 255, 255)") {
                        card.style.display = "none";
                    } else {
                        card.style.display = "flex";
                    }
                } else {
                    card.style.display = "none";
                }
            });
        } else {
            styleCards.forEach(card => {
                const cardTitle = card.getAttribute('data-title');
                const cardCategory = card.getAttribute('data-category');
                if (cardTitle.includes(searchString) && cardCategory == cat) {
                    card.style.display = "flex";
                } else {
                    card.style.display = "none";
                }
            });
        }
    }
}

function editStyle(title, img, description, prompt, promptNeggative, folder, filename,origin) {
    // title
    if (origin == "Stylez") {
        prompt = removeFirstAndLastCharacter(prompt)
        promptNeggative = removeFirstAndLastCharacter(promptNeggative)
    } else {
        prompt = decodeURIComponent(prompt).replaceAll(/%27/g, "'")
        promptNeggative = decodeURIComponent(promptNeggative.replaceAll(/%27/g, "'"))
    }
    const editorTitle = gradioApp().querySelector('#style_title_txt > label > textarea');
    applyValues(editorTitle, title);
    // img
    const imgUrlHolderElement = gradioApp().querySelector('#style_img_url_txt > label > textarea');
    applyValues(imgUrlHolderElement, img);
    // applyValues(editor_img, title);
    // description
    const editorDescription = gradioApp().querySelector('#style_description_txt > label > textarea');
    applyValues(editorDescription, description);
    // prompt
    const editorPrompt = gradioApp().querySelector('#style_prompt_txt > label > textarea');
    applyValues(editorPrompt, prompt);
    // promptNeggative
    const editorPromptNeggative = gradioApp().querySelector('#style_negative_txt > label > textarea');
    applyValues(editorPromptNeggative, promptNeggative);
    // Category
    const editorSaveFolder = gradioApp().querySelector('#style_savefolder_txt > label > div > div > div > input');
    const editorTempFolder = gradioApp().querySelector('#style_savefolder_temp > label > textarea');
    applyValues(editorTempFolder, folder);
    applyValues(editorSaveFolder, folder);
    // filename
    const editorFilename = gradioApp().querySelector('#style_filename_txt > label > textarea');
    filename = decodeURIComponent(filename); // Decode the filename
    filename = filename.replace('.json', '');
    applyValues(editorFilename, filename);
    // press tab button
    const tabsdiv = gradioApp().getElementById(`Stylez`);
    function findEditorButton() {
        const buttons = tabsdiv.querySelectorAll('button');
        for (const button of buttons) {
            if (button.innerText === '风格编辑器') {
                return button;
            }
        }
        return null;
    }
    const editorButton = findEditorButton();
    if (editorButton) {
        editorButton.click();
    }
}

function grabLastGeneratedimage() {
    const imagegallery = gradioApp().querySelector(`#${tabname}_gallery`);
    if (imagegallery) {
        const firstImage = imagegallery.querySelector('img');
        if (firstImage) {
            let imageSrc = firstImage.src;
            imageSrc = imageSrc.replace(/.*file=/, '');
            imageSrc = imageSrc.split('?')[0]; //fix for Forge
            imageSrc = decodeURIComponent(imageSrc); // Decode the URL-encoded file name
            const imgUrlHolderElement = gradioApp().querySelector('#style_img_url_txt > label > textarea');
            applyValues(imgUrlHolderElement, imageSrc);
        }
    }
}

function grabCurrentSettings() {
    // prompt
    const editorPrompt = gradioApp().querySelector('#style_prompt_txt > label > textarea');
    applyValues(editorPrompt, promptPos.value);
    // promptNeggative
    const editorPromptNeggative = gradioApp().querySelector('#style_negative_txt > label > textarea');
    applyValues(editorPromptNeggative, promptNeg.value);
}

function deleteRefresh() {
    const galleryrefresh = gradioApp().querySelector('#style_refresh');
    const stylesclear = gradioApp().querySelector('#style_clear_btn');
    galleryrefresh.click();
    stylesclear.click();
}

function saveRefresh() {
    setTimeout(() => {
        const galleryrefresh = gradioApp().querySelector('#style_refresh');
        galleryrefresh.click();
    }, 1000); // 1000 milliseconds = 1 second
}

function addFavourite(folder, filename, element) {
    let computedelem = getComputedStyle(element);
    const addfavouritebtn = gradioApp().querySelector('#stylezAddFavourite');
    const removefavouritebtn = gradioApp().querySelector('#stylezRemoveFavourite');
    filename = decodeURIComponent(filename);
    const favTempFolder = gradioApp().querySelector('#favouriteTempTxt > label > textarea');
    if (computedelem.color === "rgb(255, 255, 255)") {
        element.style.color = "#EBD617";
        applyValues(favTempFolder, folder + "/" + filename);
        addfavouritebtn.click();
    } else {
        element.style.color = "#ffffff";
        applyValues(favTempFolder, folder + "/" + filename);
        removefavouritebtn.click();
    }
}

function addQuicksave () {
    const ulElement = gradioApp().getElementById('styles_quicksave_list');
    var liElement = document.createElement('li');
    var deleteButton = document.createElement('button');
    var innerButton = document.createElement('button');
    var promptParagraph = document.createElement('button');
    var negParagraph = document.createElement('button');
    let prompt = ""
    let negprompt = ""
    if (promptPos.value !== "" || promptNeg.value !== "") {
       
        if (promptPos.value == "")
        {
            promptParagraph.disabled = true  
            promptParagraph.textContent = "EMPTY"
            prompt = "EMPTY"
        }
        else {
            promptParagraph.disabled = false
            promptParagraph.textContent = promptPos.value;
            prompt = encodeURIComponent(promptPos.value);
        }
        if (promptNeg.value == "")
        {
            negParagraph.disabled = true
            negParagraph.textContent = "EMPTY"
            negprompt = "EMPTY"
        } else {
            negParagraph.disabled = false
            negParagraph.textContent = promptNeg.value;
            negprompt = encodeURIComponent(promptNeg.value)
        }
        liElement.className = 'styles_quicksave';
        deleteButton.className = 'styles_quicksave_del';
        deleteButton.textContent = '❌';
        deleteButton.onclick = function() {
            deletequicksave(this)
        };
        promptParagraph.onclick = function() {
            applyQuickSave("pos",this.textContent)
        };
        promptParagraph.onmouseenter = function() {
            event.stopPropagation(); 
            hoverPreviewStyle(promptParagraph.textContent,negParagraph.textContent,'Quicksave');
        }
        promptParagraph.onmouseleave = function() {
            hoverPreviewStyleOut()
        }
        promptParagraph.className = 'styles_quicksave_prompt styles_quicksave_btn';

        negParagraph.onclick = function() {
            applyQuickSave("neg",this.textContent)
        };
        negParagraph.onmouseenter = function() {
            event.stopPropagation(); 
            hoverPreviewStyle(promptParagraph.textContent,negParagraph.textContent,'Quicksave');
        }
        negParagraph.onmouseleave = function() {
            hoverPreviewStyleOut()
        }
        negParagraph.className = 'styles_quicksave_neg styles_quicksave_btn';
        innerButton.className = 'styles_quicksave_apply';
        innerButton.appendChild(promptParagraph);
        innerButton.appendChild(negParagraph);
        liElement.appendChild(deleteButton);
        liElement.appendChild(innerButton);
        ulElement.appendChild(liElement);
    }
}
function applyQuickSave(box,prompt) {
    tabname = getENActiveTab();
    if (box == "pos"){
        applyValues(promptPos,prompt)
    } else {
        applyValues(promptNeg,prompt)
    }
}

function deletequicksave(elem) {
    const quicksave = elem.parentNode;
    const list = quicksave.parentNode;
    list.removeChild(quicksave);
}

function clearquicklist() {
    const list = gradioApp().getElementById("styles_quicksave_list")
    while (list.firstChild) {
        list.removeChild(list.firstChild);
    }
}

function sendToPromtbox(prompt) {
    tabname = getENActiveTab();
    promptPos = gradioApp().querySelector(`#${tabname}_prompt > label > textarea`);
    applyValues(promptPos,prompt)
}

function stylesgrabprompt() {
    tabname = getENActiveTab();
    promptPos = gradioApp().querySelector(`#${tabname}_prompt > label > textarea`);
    return promptPos.value
}

function sendToARbox(width, height) {
    // 获取txt2img的宽度和高度输入框
    var arWidthTxt2Img = gradioApp().querySelector(`#txt2img_width input`);
    var arHeightTxt2Img = gradioApp().querySelector(`#txt2img_height input`);

    // 获取img2img的宽度和高度输入框
    var arWidthImg2Img = gradioApp().querySelector(`#img2img_width input`);
    var arHeightImg2Img = gradioApp().querySelector(`#img2img_height input`);

    // 更新txt2img的宽度和高度
    if (arWidthTxt2Img && arHeightTxt2Img) {
        arWidthTxt2Img.value = width;
        arHeightTxt2Img.value = height;
        arWidthTxt2Img.dispatchEvent(new Event('input'));
        arHeightTxt2Img.dispatchEvent(new Event('input'));
    } else {
        console.error("无法找到txt2img_width或txt2img_height元素");
    }

    // 更新img2img的宽度和高度
    if (arWidthImg2Img && arHeightImg2Img) {
        arWidthImg2Img.value = width;
        arHeightImg2Img.value = height;
        arWidthImg2Img.dispatchEvent(new Event('input'));
        arHeightImg2Img.dispatchEvent(new Event('input'));
    } else {
        console.error("无法找到img2img_width或img2img_height元素");
    }
}
