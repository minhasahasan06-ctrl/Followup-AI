import axios from "axios";

// PubMed E-utilities API integration
export class PubMedService {
  private baseUrl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/";

  async search(query: string, maxResults: number = 100) {
    try {
      const response = await axios.get(`${this.baseUrl}esearch.fcgi`, {
        params: {
          db: "pubmed",
          term: query,
          retmode: "json",
          retmax: maxResults,
          usehistory: "y",
        },
      });

      const data = response.data.esearchresult;
      return {
        count: parseInt(data.count),
        ids: data.idlist || [],
        webEnv: data.webenv,
        queryKey: data.querykey,
      };
    } catch (error) {
      console.error("PubMed search error:", error);
      throw new Error("Failed to search PubMed");
    }
  }

  async fetchArticles(ids: string[]) {
    try {
      const response = await axios.get(`${this.baseUrl}esummary.fcgi`, {
        params: {
          db: "pubmed",
          id: ids.join(","),
          retmode: "json",
        },
      });

      const results = response.data.result;
      const articles = ids.map((id) => {
        const article = results[id];
        if (!article || article.error) return null;

        return {
          pmid: id,
          title: article.title || "",
          authors: article.authors?.map((a: any) => a.name) || [],
          journal: article.fulljournalname || "",
          pubDate: article.pubdate || "",
          doi: article.elocationid || "",
          abstract: article.abstract || "",
        };
      }).filter(Boolean);

      return articles;
    } catch (error) {
      console.error("PubMed fetch error:", error);
      throw new Error("Failed to fetch PubMed articles");
    }
  }
}

// PhysioNet data integration
export class PhysioNetService {
  private baseUrl = "https://physionet.org";

  async searchDatasets(query: string) {
    // PhysioNet doesn't have a public search API, so we'll provide curated dataset list
    const curatedDatasets = [
      {
        id: "mitdb",
        title: "MIT-BIH Arrhythmia Database",
        description: "48 half-hour excerpts of two-channel ambulatory ECG recordings",
        dataType: "ecg",
        recordCount: 48,
        url: "https://physionet.org/content/mitdb/1.0.0/",
        version: "1.0.0",
        license: "Open Data Commons Attribution License v1.0",
      },
      {
        id: "ptb-xl",
        title: "PTB-XL ECG Database",
        description: "Large publicly available electrocardiography dataset",
        dataType: "ecg",
        recordCount: 21837,
        url: "https://physionet.org/content/ptb-xl/1.0.3/",
        version: "1.0.3",
        license: "Creative Commons Attribution 4.0",
      },
      {
        id: "mimic-iii",
        title: "MIMIC-III Clinical Database",
        description: "Deidentified health data from ICU patients",
        dataType: "clinical_notes",
        recordCount: 46520,
        url: "https://physionet.org/content/mimiciii/1.4/",
        version: "1.4",
        license: "PhysioNet Credentialed Health Data License",
        credentialRequired: true,
      },
      {
        id: "mimic-iv",
        title: "MIMIC-IV Clinical Database",
        description: "Updated version of MIMIC-III with more recent data",
        dataType: "clinical_notes",
        recordCount: 73181,
        url: "https://physionet.org/content/mimiciv/2.2/",
        version: "2.2",
        license: "PhysioNet Credentialed Health Data License",
        credentialRequired: true,
      },
    ];

    const lowercaseQuery = query.toLowerCase();
    const filtered = curatedDatasets.filter(
      (ds) =>
        ds.title.toLowerCase().includes(lowercaseQuery) ||
        ds.description.toLowerCase().includes(lowercaseQuery) ||
        ds.dataType.toLowerCase().includes(lowercaseQuery)
    );

    return filtered;
  }

  async getDatasetInfo(datasetId: string) {
    const datasets = await this.searchDatasets("");
    return datasets.find((ds) => ds.id === datasetId);
  }
}

// Kaggle API integration
export class KaggleService {
  private username = process.env.KAGGLE_USERNAME || "";
  private key = process.env.KAGGLE_KEY || "";
  private baseUrl = "https://www.kaggle.com/api/v1";

  private getAuthHeaders() {
    const credentials = Buffer.from(`${this.username}:${this.key}`).toString("base64");
    return {
      Authorization: `Basic ${credentials}`,
    };
  }

  async searchDatasets(query: string, page: number = 1) {
    try {
      const response = await axios.get(`${this.baseUrl}/datasets/list`, {
        headers: this.getAuthHeaders(),
        params: {
          search: query,
          page,
          maxSize: 20,
        },
      });

      return response.data.map((dataset: any) => ({
        id: dataset.ref,
        owner: dataset.ref.split("/")[0],
        name: dataset.ref.split("/")[1],
        title: dataset.title,
        description: dataset.subtitle || "",
        size: dataset.size,
        downloadCount: dataset.downloadCount,
        voteCount: dataset.voteCount,
        lastUpdated: dataset.lastUpdated,
        url: `https://www.kaggle.com/datasets/${dataset.ref}`,
        license: dataset.licenseName,
      }));
    } catch (error) {
      console.error("Kaggle search error:", error);
      throw new Error("Failed to search Kaggle datasets");
    }
  }

  async getDatasetMetadata(owner: string, datasetName: string) {
    try {
      const response = await axios.get(
        `${this.baseUrl}/datasets/metadata/${owner}/${datasetName}`,
        {
          headers: this.getAuthHeaders(),
        }
      );

      return response.data;
    } catch (error) {
      console.error("Kaggle metadata error:", error);
      throw new Error("Failed to fetch Kaggle dataset metadata");
    }
  }

  async listDatasetFiles(owner: string, datasetName: string) {
    try {
      const response = await axios.get(
        `${this.baseUrl}/datasets/list/${owner}/${datasetName}`,
        {
          headers: this.getAuthHeaders(),
        }
      );

      return response.data.datasetFiles || [];
    } catch (error) {
      console.error("Kaggle list files error:", error);
      throw new Error("Failed to list Kaggle dataset files");
    }
  }

  // Note: Actual file downloads would be handled server-side
  // This method returns download URL info
  getDownloadUrl(owner: string, datasetName: string, fileName?: string) {
    if (fileName) {
      return `${this.baseUrl}/datasets/download/${owner}/${datasetName}/${fileName}`;
    }
    return `${this.baseUrl}/datasets/download/${owner}/${datasetName}`;
  }
}

// WHO Global Health Observatory (GHO) API integration
export class WHOService {
  private baseUrl = "https://ghoapi.azureedge.net/api";

  async listIndicators() {
    try {
      const response = await axios.get(`${this.baseUrl}/Indicator`, {
        params: {
          $format: "json",
        },
      });

      return response.data.value || [];
    } catch (error) {
      console.error("WHO list indicators error:", error);
      throw new Error("Failed to list WHO indicators");
    }
  }

  async searchIndicators(query: string) {
    try {
      const response = await axios.get(`${this.baseUrl}/Indicator`, {
        params: {
          $filter: `contains(IndicatorName,'${query}')`,
          $format: "json",
        },
      });

      return response.data.value || [];
    } catch (error) {
      console.error("WHO search indicators error:", error);
      throw new Error("Failed to search WHO indicators");
    }
  }

  async getIndicatorData(indicatorCode: string, filters?: {
    country?: string;
    year?: number;
    sex?: string;
  }) {
    try {
      let filterString = "";
      const filterParts: string[] = [];

      if (filters?.country) {
        filterParts.push(`SpatialDim eq '${filters.country}'`);
      }
      if (filters?.sex) {
        filterParts.push(`Dim1 eq '${filters.sex}'`);
      }
      if (filters?.year) {
        filterParts.push(`TimeDim eq ${filters.year}`);
      }

      if (filterParts.length > 0) {
        filterString = `?$filter=${filterParts.join(" and ")}`;
      }

      const response = await axios.get(
        `${this.baseUrl}/${indicatorCode}${filterString}`,
        {
          params: {
            $format: "json",
          },
        }
      );

      return response.data.value || [];
    } catch (error) {
      console.error("WHO get indicator data error:", error);
      throw new Error("Failed to fetch WHO indicator data");
    }
  }

  async getCountries() {
    try {
      const response = await axios.get(
        `${this.baseUrl}/DIMENSION/COUNTRY/DimensionValues`,
        {
          params: {
            $format: "json",
          },
        }
      );

      return response.data.value || [];
    } catch (error) {
      console.error("WHO get countries error:", error);
      throw new Error("Failed to fetch WHO countries");
    }
  }

  // Popular health indicators for immunocompromised patients
  getPopularIndicators() {
    return [
      { code: "WHOSIS_000001", name: "Life expectancy at birth" },
      { code: "HIV_0000000001", name: "HIV prevalence" },
      { code: "WHS4_543", name: "Tuberculosis incidence" },
      { code: "NCDMORT3070", name: "NCD mortality rate (30-70 years)" },
      { code: "MDG_0000000001", name: "Infant mortality rate" },
      { code: "AIR_10", name: "Ambient air pollution deaths" },
      { code: "HWF_0001", name: "Medical doctors per 10,000 population" },
      { code: "GHED_CHE_PC_PPP_SHA2011", name: "Current health expenditure per capita" },
    ];
  }
}

export const pubmedService = new PubMedService();
export const physionetService = new PhysioNetService();
export const kaggleService = new KaggleService();
export const whoService = new WHOService();
