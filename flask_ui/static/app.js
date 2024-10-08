document.addEventListener('DOMContentLoaded', function() {

    const fileInput = document.getElementById('csv-file');

    const jobTable = $('#job-table').DataTable({
        select: 'single',
        columns: [
            { data: 'id' },
            { data: 'industry_descr' },
        ]
    });
    // const jobListBody = document.getElementById('job-list-body');
    const jobDetailsContent = document.getElementById('job-details-content');

    let jobs = [];
    let codeResultsData = {};
    const codeResultsDataTable = $('#code-results-table').DataTable({
        columns: [
            { data: 'label', title: 'SIC code' },
            { data: 'description', title: 'Description' },
            { data: 'distance', title: 'Distance' },
        ],
        order: [[2, 'asc']],
        pageLength: 5,
        lengthChange: false,
        searching: false
    });

    function loadSavedData() {
        const savedData = localStorage.getItem('jobsData');
        const savedcodeResults = localStorage.getItem('codeResultsData');
        if (savedData) {
            jobs = JSON.parse(savedData);
            updateJobTable(jobs);
        }
        if (savedcodeResults) {
            codeResultsData = JSON.parse(savedcodeResults);
        }
    }

    loadSavedData();

    fileInput.addEventListener('change',handleFileSelect);

    function handleFileSelect(event) {
        localStorage.removeItem('jobsData');
        localStorage.removeItem('selectedJobId');
        const file = event.target.files[0];
        Papa.parse(file, {
            header: true,
            skipEmptyLines: true,
            error: function(error, file) { console.log("Error", error, file) },
            complete: function(results, file) {
                console.log("Complete");
                console.log(results.data);
                jobs = results.data.map((row, index) => ({
                    id: index + 1,
                    industry_descr: row.industry_descr.trim(),
                    job_title: row.job_title.trim(),//parseFloat(row.wage.trim()),
                    job_description: row.job_description.trim(),
                    sic_code: row.sic_code.trim(),
                }));
                console.log(jobs);

                // Save jobs data to localStorage
                localStorage.setItem('jobsData', JSON.stringify(jobs));
                updateJobTable(jobs);

                // Send jobs data to backend for code prediction
                // fetchcodeResults(jobs);
            }
        });
    }

    // Get the button element
    const SearchButton = document.getElementById('fetch-results');

    // Add click event listener to the button
    SearchButton.addEventListener('click', handleSearchClick);

    function handleSearchClick(event) {
        // Show loading indicator
        codeResultsDataTable.clear().draw();
        codeResultsDataTable.row
        .add({label: 'Loading...', description: 'Please wait', distance: ''}).draw();
        function delayedJob() {
            fetchcodeResults(jobs);
        }
        setTimeout(delayedJob,2000);
    }

    function fetchcodeResults(jobsData) {
        // Show loading indicator
        codeResultsDataTable.clear().draw();
        codeResultsDataTable.row
        .add({label: 'Loading...', description: 'Please wait', distance: ''}).draw();

        fetch('/predict_sic', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(jobsData),
        })
        .then(response => response.json())
        .then(response_json => {
            console.log(JSON.stringify(response_json.data));
            codeResultsData = response_json.data;
            localStorage.setItem('codeResultsData', JSON.stringify(response_json.data));
            updateCodeResultsTable(); // Redraw results table when fetch completes
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    }

    function updateJobTable(jobsData) {
        jobTable.clear();
        jobTable.rows.add(jobsData.map(job => ({
            id: job.id,
            industry_descr: job.industry_descr
        }))).draw();

        $('#job-table tbody').off('click', 'tr').on('click', 'tr', function() {
            const rowData = jobTable.row(this).data();
            localStorage.setItem('selectedJobId', rowData.id);
            const fullJobData = jobs.find(job => job.id === rowData.id);
            showJobDetails(fullJobData);
        });

        // // Restore selected row
        // const selectedJobId = localStorage.getItem('selectedJobId');
        // if (selectedJobId) {
        //     const row = jobTable.row((idx, data) => data.id === parseInt(selectedJobId));
        //     if (row.length) {
        //         row.select();
        //         const fullJobData = jobs.find(job => job.id === parseInt(selectedJobId));
        //         showJobDetails(fullJobData);
        //     }
        // }
    }

    function showJobDetails(job) {
        jobDetailsContent.innerHTML = `
            <h2>${job.industry_descr}</h2>
        `;
        showCodeResults(job.id);
    }

    function showCodeResults(jobId) {
        const results = codeResultsData.find(item => item.input_id === jobId);
        if (results && results.response) {
            results.response.forEach(row => {
                row.distance = parseFloat(row.distance).toFixed(2);
            })
            codeResultsDataTable.clear().rows.add(results.response).draw();
        } else {
            codeResultsDataTable.clear().draw();
        }
    }

    // Update code Results table when the fetch completes
    function updateCodeResultsTable() {
        const selectedJobId = localStorage.getItem('selectedJobId');
        if (selectedJobId) {
            showCodeResults(parseInt(selectedJobId));
        }
    }
});
