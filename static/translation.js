// ---------------------------------------------
// TRANSLATION DATA
// ---------------------------------------------

const translations = {
    en: {
        title: "Loneliness Prediction Survey",
        desc: "Please answer the questions honestly.",

        sec1: "Emotional Loneliness",
        sec2: "Social Interaction",
        sec3: "Daily Life",
        sec4: "Mental Health",

        q1: "I feel that I lack companionship.",
        q2: "I feel left out.",
        q3: "I feel isolated from others.",
        q4: "I have friends I can talk about important things with.",

        q9: "How often do you avoid social activities because of money?",
        q10: "How many times per week do you hang out with friends?",
        q11: "Hours per week spent outside home?",
        q12: "Weekend participation in group activities?",

        q13: "Daily phone usage?",
        q14: "Checking phone after midnight?",
        q15: "Is your sleep restful?",
        q16: "Physical activity days?",

        q17: "Feeling depressed?",
        q18: "Loss of interest?",
        q19: "What helps you feel less lonely?",

        submit: "Submit"
    },

    mn: {
        title: "Ганцаардлын Судалгаа",
        desc: "Асуултуудад шударгаар хариулна уу.",

        sec1: "Сэтгэл зүйн ганцаардал",
        sec2: "Нийгмийн харилцаа",
        sec3: "Өдөр тутмын амьдрал",
        sec4: "Сэтгэцийн эрүүл мэнд",

        q1: "Надад нөхөрлөл дутагдаж байгаа юм шиг санагддаг.",
        q2: "Би хөндийрсөн, гадуурхагдсан юм шиг санагддаг.",
        q3: "Би бусдаас тусгаарлагдсан мэт санагддаг.",
        q4: "Чухал зүйлсийн талаар ярилцаж болох найзууд надад бий.",

        q9: "Мөнгөний улмаас нийгмийн үйл ажиллагаанаас хэр олон удаа татгалздаг вэ?",
        q10: "Найзуудтайгаа долоо хоногт хэдэн удаа уулздаг вэ?",
        q11: "Долоо хоногт гэрээс гадуур хэдэн цаг өнгөрөөдөг вэ?",
        q12: "Амралтын өдрүүдээр бүлгийн үйл ажиллагаанд хэр оролцдог вэ?",

        q13: "Гар утсаа өдөрт хэр ашигладаг вэ?",
        q14: "Шөнө дунд утсаа шалгах давтамж?",
        q15: "Таны унталт амралттай байдаг уу?",
        q16: "Биеийн хөдөлгөөний давтамж?",

        q17: "Сэтгэл гуньсан үе?",
        q18: "Юмс сонирхолгүй санагдах үе?",
        q19: "Ганцаардлыг багасгахад юу тусалдаг вэ?",

        submit: "Илгээх"
    }
};


// ---------------------------------------------
// APPLY TRANSLATION FUNCTION
// ---------------------------------------------

function applyTranslation(lang) {
    document.querySelectorAll("[data-i18n]").forEach(el => {
        const key = el.dataset.i18n;
        if (translations[lang] && translations[lang][key]) {
            el.textContent = translations[lang][key];
        }
    });
}

// ---------------------------------------------
// AUTO-LOAD LANGUAGE ON PAGE LOAD
// (Optional: remembers selection)
// ---------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
    const savedLang = localStorage.getItem("lang") || "mn";
    applyTranslation(savedLang);
});

// ---------------------------------------------
// LANGUAGE SWITCHER LISTENER
// ---------------------------------------------

document.getElementById("langBtn").addEventListener("click", function () {
    let lang = this.dataset.lang === "mn" ? "en" : "mn";
    this.dataset.lang = lang;

    // Update flag + text
    document.getElementById("flagImg").src = lang === "mn" ? "flag.png" : "uk.png";
    document.getElementById("langText").textContent = lang.toUpperCase();

    // Save selection
    localStorage.setItem("lang", lang);

    // Apply translation
    applyTranslation(lang);
});
