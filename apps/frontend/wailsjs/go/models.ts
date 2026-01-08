export namespace main {
	
	export class CustomerResult {
	    customer_id: string;
	    tenure: number;
	    monthly_charges: number;
	    churn_probability: number;
	    risk_level: string;
	    action_id: number;
	    action_name: string;
	    action_cost: string;
	    reasoning: string;
	
	    static createFrom(source: any = {}) {
	        return new CustomerResult(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.customer_id = source["customer_id"];
	        this.tenure = source["tenure"];
	        this.monthly_charges = source["monthly_charges"];
	        this.churn_probability = source["churn_probability"];
	        this.risk_level = source["risk_level"];
	        this.action_id = source["action_id"];
	        this.action_name = source["action_name"];
	        this.action_cost = source["action_cost"];
	        this.reasoning = source["reasoning"];
	    }
	}
	export class DashboardStats {
	    total_customers: number;
	    high_risk: number;
	    medium_risk: number;
	    low_risk: number;
	    est_revenue_saved: number;
	    processing_time_ms: number;
	
	    static createFrom(source: any = {}) {
	        return new DashboardStats(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.total_customers = source["total_customers"];
	        this.high_risk = source["high_risk"];
	        this.medium_risk = source["medium_risk"];
	        this.low_risk = source["low_risk"];
	        this.est_revenue_saved = source["est_revenue_saved"];
	        this.processing_time_ms = source["processing_time_ms"];
	    }
	}
	export class DatasetAnalysis {
	    name: string;
	    file_path: string;
	    results: CustomerResult[];
	    stats: DashboardStats;
	    trained: boolean;
	    created_at: string;
	
	    static createFrom(source: any = {}) {
	        return new DatasetAnalysis(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.name = source["name"];
	        this.file_path = source["file_path"];
	        this.results = this.convertValues(source["results"], CustomerResult);
	        this.stats = this.convertValues(source["stats"], DashboardStats);
	        this.trained = source["trained"];
	        this.created_at = source["created_at"];
	    }
	
		convertValues(a: any, classs: any, asMap: boolean = false): any {
		    if (!a) {
		        return a;
		    }
		    if (a.slice && a.map) {
		        return (a as any[]).map(elem => this.convertValues(elem, classs));
		    } else if ("object" === typeof a) {
		        if (asMap) {
		            for (const key of Object.keys(a)) {
		                a[key] = new classs(a[key]);
		            }
		            return a;
		        }
		        return new classs(a);
		    }
		    return a;
		}
	}
	export class TrainingRecord {
	    id: number;
	    filename: string;
	    status: string;
	    accuracy: number;
	    created_at: string;
	
	    static createFrom(source: any = {}) {
	        return new TrainingRecord(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.id = source["id"];
	        this.filename = source["filename"];
	        this.status = source["status"];
	        this.accuracy = source["accuracy"];
	        this.created_at = source["created_at"];
	    }
	}

}

