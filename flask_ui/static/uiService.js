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
            ]
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
            pageLength: 5,
            lengthChange: false,
            searching: false
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
 */
export function showJobDetails(job) {
    const jobDetailsContent = document.getElementById('job-details-content');
    jobDetailsContent.innerHTML = `<h2>${job.industry_descr}</h2>`;
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
