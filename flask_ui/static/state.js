// state.js

import { ACTION_TYPES } from './actions.js';

/**
 * Reducer function to handle state changes based on actions
 * @param {Object} state - Current state
 * @param {Object} action - Action object
 * @returns {Object} New state
 */
export function reducer(state, action) {
    switch (action.type) {
      case ACTION_TYPES.LOAD_JOBS:
        return { ...state, jobs: action.payload };
      case ACTION_TYPES.SELECT_JOB:
        return { ...state, selectedJobId: action.payload };
      case ACTION_TYPES.UPDATE_RESULTS:
        return { ...state, resultsData: action.payload };
      case ACTION_TYPES.SELECT_RESULT:
        return { ...state, selectedResult: action.payload };
      case ACTION_TYPES.CLEAR_ALL:
        return initialState;
      case ACTION_TYPES.ASSIGN_RESULT:
        return {
            ...state,
            jobs: state.jobs.map(job =>
                job.id === action.payload.jobId
                    ? { ...job,
                        sic_code: action.payload.result.label,
                        sic_code_description: action.payload.result.description,
                        sic_code_score: action.payload.result.distance,
                        sic_code_rank: action.payload.result.rank,
                    }
                    : job
            )
        };
      default:
        return state;
    }
}


/**
 * Initial state of the application
 * @typedef {Object} State
 * @property {Array} jobs - List of job objects
 * @property {number|null} selectedJobId - ID of the currently selected job
 * @property {Object} resultsData - SIC code results for jobs
 */
const initialState = {
    jobs: [],
    selectedJobId: null,
    resultsData: [],
    selectedResult: {},
};

export { initialState };

/**
 * Creates a store with the given reducer function
 * @param {function} reducer - Function that returns new state based on current state and action
 * @returns {Object} Store object with getState, dispatch, and subscribe methods
 */
 export function createStore(reducer) {
    let state = initialState;
    let listeners = new Map();

    const getState = () => state;

    const dispatch = (action) => {
        state = reducer(state, action);
        if (listeners.has(action.type)) {
            listeners.get(action.type).forEach(listener => listener(state, action));
        }
        if (listeners.has('*')) {
            listeners.get('*').forEach(listener => listener(state, action));
        }
    };

    const subscribe = (actionType, listener) => {
        if (!listeners.has(actionType)) {
            listeners.set(actionType, []);
        }
        listeners.get(actionType).push(listener);
        return () => {
            if (listeners.has(actionType)) {
                listeners.set(actionType, listeners.get(actionType).filter(l => l !== listener));
                if (listeners.get(actionType).length === 0) {
                    listeners.delete(actionType);
                }
            }
        };
    };

    return { getState, dispatch, subscribe };
}
