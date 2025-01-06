document.addEventListener("DOMContentLoaded", function () {
    const loginForm = document.getElementById("loginForm");

    loginForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const email = document.getElementById("email").value;
        const password = document.getElementById("password").value;

        const response = await fetch('/login/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password }),
        });

        if (response.ok) {
            const data = await response.json();
            addBanner(data.message,"#3fad0c");
            console.log(data.message); // Handle success (e.g., redirect or show a message)

            // now redirect the user to the main page
            fadeOut();
            //window.location.href = "{% url 'home' %}"; // redirect to home page

        } else {
            const error = await response.json();
            addBanner(error.error,"#ad2b0e");
            console.error(error); // Handle errors (e.g., show an error message)
        }
    });
});

function addBanner(message,signal){
    let newNotify = $('.notify.-hidden').clone().removeClass('-hidden').appendTo('body');
    newNotify.text(message);
    newNotify.css('background',signal);
    notifyPositionCalc(newNotify);
    setTimeout(() => {
        newNotify.remove();
        notifyPositionCalc();
    }, 15000);
}

function fadeOut() {
    const loginForm = document.getElementById("loginWrapper");

    // Trigger fade-out and translate effect
    loginForm.style.transition = "opacity 2s ease, transform 2s ease";
    loginForm.style.transform = "translateY(-1000px)";
    loginForm.style.opacity = "0";

    setTimeout(() => {
        loginForm.style.display = "none";
    }, 2000); // Match the duration of the transition

    setTimeout(() => {
        window.location.href = userNextUrl; //"{{ next|escapejs }}";
    }, 2100); // Match the duration of the transition
}

notifyPositionCalc = (notifyEl) => {
    let notiFyiers = $('.notify:not(.-hidden)')
    let notifyCounter = notiFyiers.length;
    if (!notifyEl) {
        notiFyiers.map((i,v) => {
            $(v).css('top', (i - 1) * ($(v).outerHeight() + 10) + 'px');
        })
        return true;
    }

    if (notifyCounter >= 1) {
        notifyEl.css('top', (notifyCounter - 1) * (notifyEl.outerHeight() + 10) + 'px');
    }
}

$(document).ready( () => {
    $('.click-handler').on('click', () => {
        let newNotify = $('.notify.-hidden').clone().removeClass('-hidden').appendTo('body');

        notifyPositionCalc(newNotify);
        setTimeout(() => {
            newNotify.remove();
            notifyPositionCalc();
        }, 15000);
    });
});