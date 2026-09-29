# E2 P0 — task-type candidates for review

Status: **review required**. Do not use this catalog to generate states or teacher labels until the user has reviewed it.

## Scope and sources

- 500 candidates: `choice` 167, `score` 167, `noul` 166.
- Domains were expanded from the public [NAICS 2022 industry classification](https://www.census.gov/naics/) and [O*NET occupational families](https://www.onetcenter.org/taxonomy.html).
- Each candidate combines one domain with one ordinary record-based judgment. It contains no question text or options.
- Exact `type_id` values from Devset B were excluded as a name ban-list only. Gold V1 was not opened or referenced.
- `task_types_candidates.jsonl` is the machine-readable source. Remove or edit any candidate that is too technical, duplicate, or unsuitable before P1.

## Applied review decisions (2026-09-26)

- Removed all `military_operations_*` candidates.
- Removed `religious_organizations_operational_risk` and `religious_organizations_escalation_destination`.
- Kept `evidence_strength` only for legal services, insurance, law enforcement, cybersecurity operations, quality assurance, and scientific research.
- Removed `escalation_destination` globally. `action_routing` remains as the distinct choice of next action; it is not a choice of recipient.
- Removed `administrative_support` and retained human resources plus procurement as the two distinct internal-function domains.
- Retained law enforcement, fire and rescue, and emergency management: their records and response contexts differ; evidence-strength is only retained for law enforcement.
- Renamed the action candidate in banking, insurance, and accounting services to the respective case, claim, and entry resolution/disposition.
- Added narrowly scoped compliance, incident-response, and service-continuity judgments only where their domain makes them ordinary record-based decisions.
- Balanced kinds to support the required 200 / 300 split: compliance candidates use disposition or exposure where appropriate, and 11 operational follow-up candidates use follow-up urgency.

## Candidates

### 1. crop production

Source basis: NAICS/O*NET public classification.

- `crop_production_action_routing` — **choice** — Choose the next operational action that the record supports.
- `crop_production_case_status` — **choice** — Classify the current work status described in the record.
- `crop_production_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `crop_production_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `crop_production_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `crop_production_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `crop_production_followup_urgency` — **score** — Rate the urgency of the follow-up indicated by the record.
- `crop_production_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 2. animal production

Source basis: NAICS/O*NET public classification.

- `animal_production_action_routing` — **choice** — Choose the next operational action that the record supports.
- `animal_production_case_status` — **choice** — Classify the current work status described in the record.
- `animal_production_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `animal_production_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `animal_production_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `animal_production_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `animal_production_followup_urgency` — **score** — Rate the urgency of the follow-up indicated by the record.
- `animal_production_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.
- `animal_production_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 3. forestry operations

Source basis: NAICS/O*NET public classification.

- `forestry_operations_action_routing` — **choice** — Choose the next operational action that the record supports.
- `forestry_operations_case_status` — **choice** — Classify the current work status described in the record.
- `forestry_operations_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `forestry_operations_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `forestry_operations_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `forestry_operations_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `forestry_operations_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `forestry_operations_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 4. oil and gas extraction

Source basis: NAICS/O*NET public classification.

- `oil_and_gas_extraction_action_routing` — **choice** — Choose the next operational action that the record supports.
- `oil_and_gas_extraction_case_status` — **choice** — Classify the current work status described in the record.
- `oil_and_gas_extraction_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `oil_and_gas_extraction_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `oil_and_gas_extraction_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `oil_and_gas_extraction_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `oil_and_gas_extraction_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `oil_and_gas_extraction_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `oil_and_gas_extraction_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 5. mineral mining

Source basis: NAICS/O*NET public classification.

- `mineral_mining_action_routing` — **choice** — Choose the next operational action that the record supports.
- `mineral_mining_case_status` — **choice** — Classify the current work status described in the record.
- `mineral_mining_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `mineral_mining_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `mineral_mining_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `mineral_mining_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `mineral_mining_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `mineral_mining_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `mineral_mining_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 6. electric power utilities

Source basis: NAICS/O*NET public classification.

- `electric_power_utilities_action_routing` — **choice** — Choose the next operational action that the record supports.
- `electric_power_utilities_case_status` — **choice** — Classify the current work status described in the record.
- `electric_power_utilities_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `electric_power_utilities_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `electric_power_utilities_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `electric_power_utilities_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `electric_power_utilities_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `electric_power_utilities_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `electric_power_utilities_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 7. water and wastewater utilities

Source basis: NAICS/O*NET public classification.

- `water_and_wastewater_utilities_action_routing` — **choice** — Choose the next operational action that the record supports.
- `water_and_wastewater_utilities_case_status` — **choice** — Classify the current work status described in the record.
- `water_and_wastewater_utilities_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `water_and_wastewater_utilities_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `water_and_wastewater_utilities_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `water_and_wastewater_utilities_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `water_and_wastewater_utilities_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `water_and_wastewater_utilities_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `water_and_wastewater_utilities_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 8. building construction

Source basis: NAICS/O*NET public classification.

- `building_construction_action_routing` — **choice** — Choose the next operational action that the record supports.
- `building_construction_case_status` — **choice** — Classify the current work status described in the record.
- `building_construction_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `building_construction_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `building_construction_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `building_construction_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `building_construction_followup_urgency` — **score** — Rate the urgency of the follow-up indicated by the record.
- `building_construction_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `building_construction_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.
- `building_construction_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 9. heavy civil construction

Source basis: NAICS/O*NET public classification.

- `heavy_civil_construction_action_routing` — **choice** — Choose the next operational action that the record supports.
- `heavy_civil_construction_case_status` — **choice** — Classify the current work status described in the record.
- `heavy_civil_construction_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `heavy_civil_construction_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `heavy_civil_construction_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `heavy_civil_construction_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `heavy_civil_construction_followup_urgency` — **score** — Rate the urgency of the follow-up indicated by the record.
- `heavy_civil_construction_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `heavy_civil_construction_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 10. specialty trade construction

Source basis: NAICS/O*NET public classification.

- `specialty_trade_construction_action_routing` — **choice** — Choose the next operational action that the record supports.
- `specialty_trade_construction_case_status` — **choice** — Classify the current work status described in the record.
- `specialty_trade_construction_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `specialty_trade_construction_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `specialty_trade_construction_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `specialty_trade_construction_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `specialty_trade_construction_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `specialty_trade_construction_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `specialty_trade_construction_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 11. food manufacturing

Source basis: NAICS/O*NET public classification.

- `food_manufacturing_action_routing` — **choice** — Choose the next operational action that the record supports.
- `food_manufacturing_case_status` — **choice** — Classify the current work status described in the record.
- `food_manufacturing_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `food_manufacturing_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `food_manufacturing_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `food_manufacturing_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `food_manufacturing_followup_urgency` — **score** — Rate the urgency of the follow-up indicated by the record.
- `food_manufacturing_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `food_manufacturing_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.
- `food_manufacturing_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 12. chemical manufacturing

Source basis: NAICS/O*NET public classification.

- `chemical_manufacturing_action_routing` — **choice** — Choose the next operational action that the record supports.
- `chemical_manufacturing_case_status` — **choice** — Classify the current work status described in the record.
- `chemical_manufacturing_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `chemical_manufacturing_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `chemical_manufacturing_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `chemical_manufacturing_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `chemical_manufacturing_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `chemical_manufacturing_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `chemical_manufacturing_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 13. metal fabrication

Source basis: NAICS/O*NET public classification.

- `metal_fabrication_action_routing` — **choice** — Choose the next operational action that the record supports.
- `metal_fabrication_case_status` — **choice** — Classify the current work status described in the record.
- `metal_fabrication_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `metal_fabrication_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `metal_fabrication_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `metal_fabrication_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `metal_fabrication_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `metal_fabrication_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `metal_fabrication_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 14. computer and electronics manufacturing

Source basis: NAICS/O*NET public classification.

- `computer_electronics_manufacturing_action_routing` — **choice** — Choose the next operational action that the record supports.
- `computer_electronics_manufacturing_case_status` — **choice** — Classify the current work status described in the record.
- `computer_electronics_manufacturing_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `computer_electronics_manufacturing_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `computer_electronics_manufacturing_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `computer_electronics_manufacturing_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `computer_electronics_manufacturing_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `computer_electronics_manufacturing_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `computer_electronics_manufacturing_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 15. transport equipment manufacturing

Source basis: NAICS/O*NET public classification.

- `transport_equipment_manufacturing_action_routing` — **choice** — Choose the next operational action that the record supports.
- `transport_equipment_manufacturing_case_status` — **choice** — Classify the current work status described in the record.
- `transport_equipment_manufacturing_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `transport_equipment_manufacturing_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `transport_equipment_manufacturing_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `transport_equipment_manufacturing_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `transport_equipment_manufacturing_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `transport_equipment_manufacturing_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `transport_equipment_manufacturing_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.
- `transport_equipment_manufacturing_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 16. warehousing

Source basis: NAICS/O*NET public classification.

- `warehousing_action_routing` — **choice** — Choose the next operational action that the record supports.
- `warehousing_case_status` — **choice** — Classify the current work status described in the record.
- `warehousing_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `warehousing_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `warehousing_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `warehousing_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `warehousing_followup_urgency` — **score** — Rate the urgency of the follow-up indicated by the record.
- `warehousing_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.
- `warehousing_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 17. air transportation

Source basis: NAICS/O*NET public classification.

- `air_transportation_action_routing` — **choice** — Choose the next operational action that the record supports.
- `air_transportation_case_status` — **choice** — Classify the current work status described in the record.
- `air_transportation_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `air_transportation_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `air_transportation_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `air_transportation_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `air_transportation_followup_urgency` — **score** — Rate the urgency of the follow-up indicated by the record.
- `air_transportation_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `air_transportation_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.
- `air_transportation_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 18. rail transportation

Source basis: NAICS/O*NET public classification.

- `rail_transportation_action_routing` — **choice** — Choose the next operational action that the record supports.
- `rail_transportation_case_status` — **choice** — Classify the current work status described in the record.
- `rail_transportation_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `rail_transportation_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `rail_transportation_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `rail_transportation_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `rail_transportation_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `rail_transportation_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `rail_transportation_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.
- `rail_transportation_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 19. water transportation

Source basis: NAICS/O*NET public classification.

- `water_transportation_action_routing` — **choice** — Choose the next operational action that the record supports.
- `water_transportation_case_status` — **choice** — Classify the current work status described in the record.
- `water_transportation_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `water_transportation_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `water_transportation_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `water_transportation_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `water_transportation_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `water_transportation_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `water_transportation_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 20. truck transportation

Source basis: NAICS/O*NET public classification.

- `truck_transportation_action_routing` — **choice** — Choose the next operational action that the record supports.
- `truck_transportation_case_status` — **choice** — Classify the current work status described in the record.
- `truck_transportation_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `truck_transportation_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `truck_transportation_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `truck_transportation_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `truck_transportation_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `truck_transportation_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `truck_transportation_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.
- `truck_transportation_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 21. transit and ground transportation

Source basis: NAICS/O*NET public classification.

- `transit_and_ground_transportation_action_routing` — **choice** — Choose the next operational action that the record supports.
- `transit_and_ground_transportation_case_status` — **choice** — Classify the current work status described in the record.
- `transit_and_ground_transportation_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `transit_and_ground_transportation_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `transit_and_ground_transportation_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `transit_and_ground_transportation_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `transit_and_ground_transportation_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `transit_and_ground_transportation_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `transit_and_ground_transportation_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 22. courier delivery

Source basis: NAICS/O*NET public classification.

- `courier_delivery_action_routing` — **choice** — Choose the next operational action that the record supports.
- `courier_delivery_case_status` — **choice** — Classify the current work status described in the record.
- `courier_delivery_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `courier_delivery_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `courier_delivery_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `courier_delivery_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `courier_delivery_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `courier_delivery_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `courier_delivery_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 23. wholesale trade

Source basis: NAICS/O*NET public classification.

- `wholesale_trade_action_routing` — **choice** — Choose the next operational action that the record supports.
- `wholesale_trade_case_status` — **choice** — Classify the current work status described in the record.
- `wholesale_trade_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `wholesale_trade_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `wholesale_trade_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `wholesale_trade_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `wholesale_trade_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 24. grocery retail

Source basis: NAICS/O*NET public classification.

- `grocery_retail_action_routing` — **choice** — Choose the next operational action that the record supports.
- `grocery_retail_case_status` — **choice** — Classify the current work status described in the record.
- `grocery_retail_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `grocery_retail_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `grocery_retail_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `grocery_retail_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `grocery_retail_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 25. motor vehicle retail

Source basis: NAICS/O*NET public classification.

- `motor_vehicle_retail_action_routing` — **choice** — Choose the next operational action that the record supports.
- `motor_vehicle_retail_case_status` — **choice** — Classify the current work status described in the record.
- `motor_vehicle_retail_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `motor_vehicle_retail_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `motor_vehicle_retail_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `motor_vehicle_retail_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `motor_vehicle_retail_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 26. accommodation

Source basis: NAICS/O*NET public classification.

- `accommodation_action_routing` — **choice** — Choose the next operational action that the record supports.
- `accommodation_case_status` — **choice** — Classify the current work status described in the record.
- `accommodation_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `accommodation_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `accommodation_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `accommodation_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `accommodation_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 27. food service

Source basis: NAICS/O*NET public classification.

- `food_service_action_routing` — **choice** — Choose the next operational action that the record supports.
- `food_service_case_status` — **choice** — Classify the current work status described in the record.
- `food_service_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `food_service_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `food_service_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `food_service_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `food_service_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 28. publishing

Source basis: NAICS/O*NET public classification.

- `publishing_action_routing` — **choice** — Choose the next operational action that the record supports.
- `publishing_case_status` — **choice** — Classify the current work status described in the record.
- `publishing_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `publishing_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `publishing_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `publishing_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `publishing_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 29. telecommunications

Source basis: NAICS/O*NET public classification.

- `telecommunications_action_routing` — **choice** — Choose the next operational action that the record supports.
- `telecommunications_case_status` — **choice** — Classify the current work status described in the record.
- `telecommunications_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `telecommunications_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `telecommunications_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `telecommunications_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `telecommunications_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `telecommunications_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 30. software services

Source basis: NAICS/O*NET public classification.

- `software_services_action_routing` — **choice** — Choose the next operational action that the record supports.
- `software_services_case_status` — **choice** — Classify the current work status described in the record.
- `software_services_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `software_services_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `software_services_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `software_services_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `software_services_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `software_services_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 31. banking

Source basis: NAICS/O*NET public classification.

- `banking_case_disposition` — **choice** — Choose the record-supported disposition of the banking case.
- `banking_case_status` — **choice** — Classify the current work status described in the record.
- `banking_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `banking_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `banking_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `banking_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `banking_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `banking_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `banking_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 32. insurance

Source basis: NAICS/O*NET public classification.

- `insurance_claim_resolution` — **choice** — Choose the record-supported resolution of the insurance claim.
- `insurance_case_status` — **choice** — Classify the current work status described in the record.
- `insurance_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `insurance_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `insurance_evidence_strength` — **score** — Rate how strongly the record supports the proposed conclusion.
- `insurance_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `insurance_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `insurance_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `insurance_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `insurance_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 33. real estate

Source basis: NAICS/O*NET public classification.

- `real_estate_action_routing` — **choice** — Choose the next operational action that the record supports.
- `real_estate_case_status` — **choice** — Classify the current work status described in the record.
- `real_estate_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `real_estate_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `real_estate_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `real_estate_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `real_estate_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 34. legal services

Source basis: NAICS/O*NET public classification.

- `legal_services_action_routing` — **choice** — Choose the next operational action that the record supports.
- `legal_services_case_status` — **choice** — Classify the current work status described in the record.
- `legal_services_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `legal_services_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `legal_services_evidence_strength` — **score** — Rate how strongly the record supports the proposed conclusion.
- `legal_services_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `legal_services_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `legal_services_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `legal_services_compliance_exposure` — **score** — Rate the compliance exposure indicated by the record.

### 35. accounting services

Source basis: NAICS/O*NET public classification.

- `accounting_services_case_status` — **choice** — Classify the current work status described in the record.
- `accounting_services_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `accounting_services_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `accounting_services_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `accounting_services_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `accounting_services_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 36. engineering services

Source basis: NAICS/O*NET public classification.

- `engineering_services_action_routing` — **choice** — Choose the next operational action that the record supports.
- `engineering_services_case_status` — **choice** — Classify the current work status described in the record.
- `engineering_services_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `engineering_services_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `engineering_services_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `engineering_services_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `engineering_services_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 37. scientific research

Source basis: NAICS/O*NET public classification.

- `scientific_research_action_routing` — **choice** — Choose the next operational action that the record supports.
- `scientific_research_case_status` — **choice** — Classify the current work status described in the record.
- `scientific_research_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `scientific_research_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `scientific_research_evidence_strength` — **score** — Rate how strongly the record supports the proposed conclusion.
- `scientific_research_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `scientific_research_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `scientific_research_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 38. waste remediation

Source basis: NAICS/O*NET public classification.

- `waste_remediation_action_routing` — **choice** — Choose the next operational action that the record supports.
- `waste_remediation_case_status` — **choice** — Classify the current work status described in the record.
- `waste_remediation_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `waste_remediation_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `waste_remediation_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `waste_remediation_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `waste_remediation_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `waste_remediation_compliance_exposure` — **score** — Rate the compliance exposure indicated by the record.
- `waste_remediation_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 39. elementary education

Source basis: NAICS/O*NET public classification.

- `elementary_education_action_routing` — **choice** — Choose the next operational action that the record supports.
- `elementary_education_case_status` — **choice** — Classify the current work status described in the record.
- `elementary_education_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `elementary_education_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `elementary_education_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `elementary_education_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `elementary_education_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 40. higher education

Source basis: NAICS/O*NET public classification.

- `higher_education_action_routing` — **choice** — Choose the next operational action that the record supports.
- `higher_education_case_status` — **choice** — Classify the current work status described in the record.
- `higher_education_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `higher_education_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `higher_education_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `higher_education_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `higher_education_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `higher_education_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 41. outpatient healthcare

Source basis: NAICS/O*NET public classification.

- `outpatient_healthcare_action_routing` — **choice** — Choose the next operational action that the record supports.
- `outpatient_healthcare_case_status` — **choice** — Classify the current work status described in the record.
- `outpatient_healthcare_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `outpatient_healthcare_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `outpatient_healthcare_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `outpatient_healthcare_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `outpatient_healthcare_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `outpatient_healthcare_compliance_disposition` — **choice** — Classify the compliance disposition supported by the record.
- `outpatient_healthcare_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 42. hospital care

Source basis: NAICS/O*NET public classification.

- `hospital_care_action_routing` — **choice** — Choose the next operational action that the record supports.
- `hospital_care_case_status` — **choice** — Classify the current work status described in the record.
- `hospital_care_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `hospital_care_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `hospital_care_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `hospital_care_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `hospital_care_followup_urgency` — **score** — Rate the urgency of the follow-up indicated by the record.
- `hospital_care_compliance_exposure` — **score** — Rate the compliance exposure indicated by the record.
- `hospital_care_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.
- `hospital_care_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 43. residential care

Source basis: NAICS/O*NET public classification.

- `residential_care_action_routing` — **choice** — Choose the next operational action that the record supports.
- `residential_care_case_status` — **choice** — Classify the current work status described in the record.
- `residential_care_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `residential_care_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `residential_care_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `residential_care_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `residential_care_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `residential_care_compliance_exposure` — **score** — Rate the compliance exposure indicated by the record.
- `residential_care_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.
- `residential_care_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 44. social assistance

Source basis: NAICS/O*NET public classification.

- `social_assistance_action_routing` — **choice** — Choose the next operational action that the record supports.
- `social_assistance_case_status` — **choice** — Classify the current work status described in the record.
- `social_assistance_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `social_assistance_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `social_assistance_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `social_assistance_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `social_assistance_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `social_assistance_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.
- `social_assistance_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 45. performing arts

Source basis: NAICS/O*NET public classification.

- `performing_arts_action_routing` — **choice** — Choose the next operational action that the record supports.
- `performing_arts_case_status` — **choice** — Classify the current work status described in the record.
- `performing_arts_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `performing_arts_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `performing_arts_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `performing_arts_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `performing_arts_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `performing_arts_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 46. museums and parks

Source basis: NAICS/O*NET public classification.

- `museums_and_parks_action_routing` — **choice** — Choose the next operational action that the record supports.
- `museums_and_parks_case_status` — **choice** — Classify the current work status described in the record.
- `museums_and_parks_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `museums_and_parks_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `museums_and_parks_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `museums_and_parks_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `museums_and_parks_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `museums_and_parks_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 47. fitness and recreation

Source basis: NAICS/O*NET public classification.

- `fitness_and_recreation_action_routing` — **choice** — Choose the next operational action that the record supports.
- `fitness_and_recreation_case_status` — **choice** — Classify the current work status described in the record.
- `fitness_and_recreation_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `fitness_and_recreation_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `fitness_and_recreation_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `fitness_and_recreation_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `fitness_and_recreation_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 48. repair and maintenance

Source basis: NAICS/O*NET public classification.

- `repair_and_maintenance_action_routing` — **choice** — Choose the next operational action that the record supports.
- `repair_and_maintenance_case_status` — **choice** — Classify the current work status described in the record.
- `repair_and_maintenance_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `repair_and_maintenance_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `repair_and_maintenance_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `repair_and_maintenance_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `repair_and_maintenance_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `repair_and_maintenance_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 49. personal care services

Source basis: NAICS/O*NET public classification.

- `personal_care_services_action_routing` — **choice** — Choose the next operational action that the record supports.
- `personal_care_services_case_status` — **choice** — Classify the current work status described in the record.
- `personal_care_services_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `personal_care_services_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `personal_care_services_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `personal_care_services_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `personal_care_services_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 50. religious organizations

Source basis: NAICS/O*NET public classification.

- `religious_organizations_action_routing` — **choice** — Choose the next operational action that the record supports.
- `religious_organizations_case_status` — **choice** — Classify the current work status described in the record.
- `religious_organizations_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `religious_organizations_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `religious_organizations_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `religious_organizations_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 51. public administration

Source basis: NAICS/O*NET public classification.

- `public_administration_action_routing` — **choice** — Choose the next operational action that the record supports.
- `public_administration_case_status` — **choice** — Classify the current work status described in the record.
- `public_administration_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `public_administration_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `public_administration_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `public_administration_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `public_administration_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `public_administration_compliance_exposure` — **score** — Rate the compliance exposure indicated by the record.
- `public_administration_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 52. emergency management

Source basis: NAICS/O*NET public classification.

- `emergency_management_action_routing` — **choice** — Choose the next operational action that the record supports.
- `emergency_management_case_status` — **choice** — Classify the current work status described in the record.
- `emergency_management_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `emergency_management_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `emergency_management_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `emergency_management_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `emergency_management_followup_urgency` — **score** — Rate the urgency of the follow-up indicated by the record.
- `emergency_management_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.
- `emergency_management_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 53. law enforcement

Source basis: NAICS/O*NET public classification.

- `law_enforcement_action_routing` — **choice** — Choose the next operational action that the record supports.
- `law_enforcement_case_status` — **choice** — Classify the current work status described in the record.
- `law_enforcement_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `law_enforcement_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `law_enforcement_evidence_strength` — **score** — Rate how strongly the record supports the proposed conclusion.
- `law_enforcement_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `law_enforcement_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `law_enforcement_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `law_enforcement_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 54. fire and rescue

Source basis: NAICS/O*NET public classification.

- `fire_and_rescue_action_routing` — **choice** — Choose the next operational action that the record supports.
- `fire_and_rescue_case_status` — **choice** — Classify the current work status described in the record.
- `fire_and_rescue_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `fire_and_rescue_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `fire_and_rescue_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `fire_and_rescue_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `fire_and_rescue_followup_urgency` — **score** — Rate the urgency of the follow-up indicated by the record.
- `fire_and_rescue_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.
- `fire_and_rescue_service_continuity_impact` — **score** — Rate the impact on continued service or operations indicated by the record.

### 55. human resources

Source basis: NAICS/O*NET public classification.

- `human_resources_action_routing` — **choice** — Choose the next operational action that the record supports.
- `human_resources_case_status` — **choice** — Classify the current work status described in the record.
- `human_resources_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `human_resources_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `human_resources_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `human_resources_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `human_resources_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 56. procurement

Source basis: NAICS/O*NET public classification.

- `procurement_action_routing` — **choice** — Choose the next operational action that the record supports.
- `procurement_case_status` — **choice** — Classify the current work status described in the record.
- `procurement_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `procurement_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `procurement_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `procurement_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `procurement_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.

### 57. quality assurance

Source basis: NAICS/O*NET public classification.

- `quality_assurance_action_routing` — **choice** — Choose the next operational action that the record supports.
- `quality_assurance_case_status` — **choice** — Classify the current work status described in the record.
- `quality_assurance_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `quality_assurance_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `quality_assurance_evidence_strength` — **score** — Rate how strongly the record supports the proposed conclusion.
- `quality_assurance_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `quality_assurance_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `quality_assurance_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `quality_assurance_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 58. cybersecurity operations

Source basis: NAICS/O*NET public classification.

- `cybersecurity_operations_action_routing` — **choice** — Choose the next operational action that the record supports.
- `cybersecurity_operations_case_status` — **choice** — Classify the current work status described in the record.
- `cybersecurity_operations_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `cybersecurity_operations_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `cybersecurity_operations_evidence_strength` — **score** — Rate how strongly the record supports the proposed conclusion.
- `cybersecurity_operations_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `cybersecurity_operations_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `cybersecurity_operations_followup_urgency` — **score** — Rate the urgency of the follow-up indicated by the record.
- `cybersecurity_operations_compliance_exposure` — **score** — Rate the compliance exposure indicated by the record.
- `cybersecurity_operations_incident_response_category` — **choice** — Classify the response category required by the incident described in the record.

### 59. environmental compliance

Source basis: NAICS/O*NET public classification.

- `environmental_compliance_action_routing` — **choice** — Choose the next operational action that the record supports.
- `environmental_compliance_case_status` — **choice** — Classify the current work status described in the record.
- `environmental_compliance_priority_level` — **score** — Rate the urgency of action indicated by the record.
- `environmental_compliance_operational_risk` — **score** — Rate the operational risk indicated by the record.
- `environmental_compliance_required_condition_present` — **noul** — Decide whether the record states that a required condition is present.
- `environmental_compliance_procedure_followed` — **noul** — Decide whether the record states that the relevant procedure was followed.
- `environmental_compliance_followup_required` — **noul** — Decide whether the record indicates that follow-up action is required.
- `environmental_compliance_compliance_exposure` — **score** — Rate the compliance exposure indicated by the record.

