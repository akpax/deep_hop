$(document).ready(function () {
    $("form").on("submit", function (event) {
        event.preventDefault();
        $("input[type='submit']").prop("disabled", true);
        $(this).hide();
        $("#loading").show();

        $.ajax({
            url: "/load_model",
            method: "POST",
            success: function (response) {
                if (response.message === "Model loaded successfully") {
                    console.log("Model loaded successfully");
                    // Redirect only after the model has successfully loaded
                    window.location.href = "/generate_lyrics";
                } else {
                    console.log("Model is already loaded");
                    window.location.href = "/generate_lyrics";
                }
            },
            error: function (error) {
                console.error("Error loading model:", error.responseText);
                // Handle the error, perhaps by re-enabling the form or showing an error message
            },
        });
    });
});
