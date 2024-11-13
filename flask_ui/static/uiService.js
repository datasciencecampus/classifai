/**
 * @file uiService.js
 * @description Provides UI-related functions for the SIC/SOC Coding Tool.
 */
import { getResultsForJob } from './dataService.js';

/**
 * Initializes DataTables for jobs and code results.
 * @returns {{jobTable: DataTables.Api, resultsDataTable: DataTables.Api}} The
 * initialized DataTables.
 */
export function initTables() {
    const jobTable = new DataTable(
        document.getElementById('job-table'),
        {
            select: 'single',
            columns: [
                { data: 'id', title: 'ID' },
                { data: 'industry_description', title: 'Industry Description' },
                { data: 'sic_code', title: 'SIC Code' }
            ],
            order: [[0,'asc']],
            pageLength: 25,
            lengthChange: false,
        }
    );

    const resultsDataTable = new DataTable(
        document.getElementById('results-table'),
        {
            select: 'single',
            columns: [
                { data: 'label', title: 'SIC code' },
                { data: 'description', title: 'Description' },
                { data: 'distance', title: 'Distance' },
            ],
            order: [[2, 'asc']],
            pageLength: 15,
            lengthChange: false,
            searching: false,
            columnDefs: [{targets: [0,2], width: '20%'}]
        }
    );

    return { jobTable, resultsDataTable };
}

/**
 * Updates the job table with new data.
 * @param {Array} jobsData - The array of job objects.
 * @param {DataTables.Api} jobTable - The job DataTable instance.
 */
export function populateJobTable(jobsData, jobTable) {
    jobTable.clear();
    jobTable.rows.add(jobsData.map(job => ({
        id: job.id,
        industry_description: job.industry_description,
        sic_code: job.sic_code
    }))).draw(false);
}

/**
 * Displays job details in the UI above the results table.
 * @param {Object} job - The job object to display.
 * @param {Function} onSave - Callback function when saving edited description
 */
export function showJobDetails(job, onSave) {
    const jobDetailsContent = document.getElementById('job-details');

    // Flag to track if we're in edit mode
    let isEditing = false;

    function createViewMode(job) {
        return `
            <div class="job-header">
                <h2>${job.industry_description || 'Select a row...'}</h2>
                ${job.industry_description ? `
                    <button class="edit-btn" id="edit-button">Edit</button>
                ` : ''}
            </div>
            <p>Assigned code: ${job.sic_code || 'None'}</p>
            <p>Assigned description: ${job.sic_code_description || 'None'}</p>
        `;
    }

    function createEditMode(job) {
        return `
            <div class="job-header">
                <input
                    type="text"
                    id="edit-description"
                    value="${job.industry_description || ''}"
                    class="edit-input"
                >
                <div class="edit-buttons">
                    <button class="ok-btn" id="ok-button">Ok</button>
                    <button class="cancel-btn" id="cancel-button">Cancel</button>
                </div>
            </div>
            <p>Assigned code: ${job.sic_code || 'None'}</p>
            <p>Assigned description: ${job.sic_code_description || 'None'}</p>
        `;
    }

    function render(job) {
        jobDetailsContent.innerHTML = isEditing ? createEditMode(job) : createViewMode(job);

        if (isEditing) {
            // Add event listeners for edit mode
            const okButton = document.getElementById('ok-button');
            const cancelButton = document.getElementById('cancel-button');
            const inputField = document.getElementById('edit-description');

            okButton.addEventListener('click', () => {
                const newDescription = inputField.value.trim();
                if (newDescription !== job.industry_description) {
                    job = { ...job, industry_description: newDescription};
                    // Call the provided callback with updated value
                    onSave(job);
                }
                isEditing = false;
                render(job);
            });

            cancelButton.addEventListener('click', () => {
                isEditing = false;
                render(job);
            });
        } else {
            // Add event listener for view mode
            const editButton = document.getElementById('edit-button');
            if (editButton) {
                editButton.addEventListener('click', () => {
                    isEditing = true;
                    render(job);
                });
            }
        }
    }

    // Initial render
    render(job);
}


/**
 * Updates the code results table for the currently selected job.
 * @param {number} selectedJobId - The ID of the job to show results for.
 * @param {Array} resultsData - the full results data.
 * @param {DataTables.Api} resultsDataTable - the results DataTable instance
 */
export function updateResultsTable(selectedJobId,resultsData,resultsDataTable) {

    try {
        resultsDataTable.clear();
        if (selectedJobId && resultsData) {
            const results = getResultsForJob(selectedJobId, resultsData);
            if (results && results.length > 0) {
                resultsDataTable.rows.add(results);
            }
        }
        resultsDataTable.draw(false);

        // Select and click the first row
        if (resultsDataTable.rows().count() > 0) {
            resultsDataTable.row(0).select();

            // Get the DOM node of the first row and trigger a click event
            const firstRow = resultsDataTable.row(0).node();
            if (firstRow) {
                firstRow.dispatchEvent(new Event('click', { bubbles: true }));
            }
        }
    } catch (error) {
        console.error('Error in updateResultsTable:', error);
        resultsDataTable.clear().draw();
    }
}
