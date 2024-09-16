document.addEventListener('DOMContentLoaded', function() {

    const fileInput = document.getElementById('csv-file');

    const jobTable = $('#job-table').DataTable({
        select: 'single',
        columns: [
            { data: 'id' },
            { data: 'title' },
            { data: 'description' }
        ]
    });
    const jobListBody = document.getElementById('job-list-body');
    const jobDetailsContent = document.getElementById('job-details-content');
    const socResultsTable = document.getElementById('soc-results-table');

    function loadSavedData() {
        const savedData = localStorage.getItem('jobsData');
        if (savedData) {
            const jobs = JSON.parse(savedData);
            updateJobTable(jobs);
        }
    }

    loadSavedData()

    fileInput.addEventListener('change',handleFileSelect);

    function handleFileSelect(event) {
        localStorage.removeItem('jobsData');
        localStorage.removeItem('selectedJobId');
        const file = event.target.files[0];
        Papa.parse(file, {
            header: true,
            skipEmptyLines: true,
            error: function(error,file){ console.log("Error",error,file)},
            complete: function(results,file) {
                console.log("Complete")
                console.log(results.data)
                const jobs = results.data
                .map((row, index) => ({
                    id: index + 1,
                    title: row.title.trim(),
                    description: row.description.trim(),
                    wage: parseFloat(row.wage.trim()),
                    supervision: row.supervision.trim(),
                    employer: row.employer.trim(),
                }));
                console.log(jobs)

                // Save jobs data to localStorage
                localStorage.setItem('jobsData', JSON.stringify(jobs));
                updateJobTable(jobs);
            }
        });
    }

    function updateJobTable(jobs) {
        jobTable.clear();
        jobTable.rows.add(jobs.map(job => ({
            id: job.id,
            title: job.title,
            description: job.description
        }))).draw();

        $('#job-table tbody').on('click', 'tr', function() {
            const rowData = jobTable.row(this).data();
            localStorage.setItem('selectedJobId', rowData.id);
            const fullJobData = jobs.find(job => job.id === rowData.id);
            showJobDetails(fullJobData);
        });

        // Restore selected row
        const selectedJobId = localStorage.getItem('selectedJobId');
        if (selectedJobId) {
            const row = jobTable.row((idx, data) => data.id === parseInt(selectedJobId));
            if (row.length) {
                row.select();
                const fullJobData = jobs.find(job => job.id === parseInt(selectedJobId));
                showJobDetails(fullJobData);
            }
        }
    }

    function showJobDetails(job) {
        jobDetailsContent.innerHTML = `
            <h2>${job.title}</h2>
            <p><strong>Description:</strong> ${job.description}</p>
            <p><strong>Wage:</strong> ${job.wage.toLocaleString('en-UK', { style: 'currency', currency: 'GBP' })}</p>
            <p><strong>Supervisory responsibilities:</strong> ${job.supervision}</p>
            <p><strong>Employer:</strong> ${job.employer}</p>

        `;
        showMockSocResults();
    }

    function showMockSocResults() {
        socResultsTable.innerHTML = `
            <tr>
                <th>SOC code</th>
                <th>Description</th>
                <th>Confidence</th>
            </tr>
            <tr>
                <td>34215</td>
                <td>Professional administrators in export</td>
                <td>0.90</td>
            </tr>
            <tr>
                <td>16184</td>
                <td>Sales and support professionals in import</td>
                <td>0.38</td>
            </tr>
            <tr>
                <td>92845</td>
                <td>Administrative occupations n.e.c.</td>
                <td>0.34</td>
            </tr>
            <tr>
                <td>09101</td>
                <td>Accounts payable administrators</td>
                <td>0.12</td>
            </tr>
        `;
    }
});
