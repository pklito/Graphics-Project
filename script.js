fetch('README.md')
    .then(response => response.text())
    .then(text => {
        document.getElementById('content').innerHTML = marked(text);
    })
    .catch(error => console.error('Error fetching README.md:', error));
