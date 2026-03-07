propositions
  safe, clean, charged
end propositions

alphabet
  powerset()
end alphabet

options
  semantics = MaxAE
  auto-complete = minimal
end options

ltlf-formulas
  safety: G safe
  liveness: F clean
  charge: charged U clean
end ltlf-formulas

preferences
  safety > liveness >= charge
end preferences
