"""Scientific invariants: exact equivalence, sequential evidence, isolation, resume."""
import dataclasses, json, math, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
import discovery as d

torch.set_num_threads(1)

class ScientificTests(unittest.TestCase):
    def test_exact_equivalence_is_restriction_dependent(self):
        restricted=d.make_catalog('kinetics','restricted'); full=d.make_catalog('kinetics','full')
        def idx(b,f): return next(i for i,m in enumerate(b['models']) if m['family']==f and m['a']=='1' and m['b']=='1')
        i,j=idx(restricted,'michaelis_menten'),idx(restricted,'competitive')
        self.assertEqual(restricted['exact'][i],restricted['exact'][j])
        self.assertEqual(restricted['groups'][i],restricted['groups'][j])
        self.assertNotEqual(full['exact'][idx(full,'michaelis_menten')],full['exact'][idx(full,'competitive')])

    def test_no_duplicate_global_atoms_and_prior_normalized(self):
        for dom in ('mechanics','kinetics'):
            b=d.make_catalog(dom)
            self.assertEqual(len(b['exact']),len(set(b['exact'])))
            self.assertAlmostEqual(sum(b['prior']),1.)
            self.assertTrue(np.isfinite(b['mu']).all())

    def test_log_evidence_matches_independent_likelihood_product(self):
        prior=torch.tensor([.2,.3,.5],dtype=torch.float64)
        inc=torch.tensor([[[-.4,-1.2,-.9]],[[-1.2,-.1,-.8]],[[-.7,-.8,-.1]]],dtype=torch.float64)
        ll=torch.zeros((1,3),dtype=torch.float64); mx=ll.clone(); likelihood=np.ones(3); refmax=np.ones(3)
        for step in inc:
            ll,mx=d.update_evidence(ll,mx,step,prior)
            likelihood*=np.exp(step.numpy()[0]); q=(prior.numpy()*likelihood).sum()
            refmax=np.maximum(refmax,q/likelihood)
            np.testing.assert_allclose(mx.numpy()[0],np.log(refmax),atol=1e-13)

    def test_identical_distributions_cannot_be_separated_by_safe_rule(self):
        b=d.make_catalog('kinetics','restricted'); t=d.tensors(b,'cpu')
        cfg=d.Config(alpha=.05)
        ll=torch.zeros((1,len(b['models'])),dtype=torch.float64); mx=ll.clone()
        true=next(i for i,m in enumerate(b['models']) if m['family']=='competitive' and m['a']=='1' and m['b']=='1')
        for _ in range(40):
            for a in range(len(b['actions'])):
                y=t['mu'][true,a]
                ll,mx=d.update_evidence(ll,mx,-.5*((y-t['mu'][:,a])/.01).square()[None,:],t['prior'])
        kind,ans=d.declaration(ll,mx,t,cfg,'safe')
        self.assertEqual(int(kind[0]),2)
        self.assertEqual(int(ans[0]),int(b['groups'][true]))

    def test_policies_respect_cost_mask(self):
        b=d.make_catalog('mechanics'); t=d.tensors(b,'cpu'); m=len(b['models'])
        ll=torch.zeros((3,m),dtype=torch.float64); mx=ll.clone()
        allowed=(t['cost']<=1).expand(3,-1)
        for policy in ('random','variance','eig','chernoff','deficit'):
            a=d.choose_action(policy,ll,mx,t,.15,allowed,torch.tensor([.1,.5,.9],dtype=torch.float64),
                torch.full((3,4),.3,dtype=torch.float64),torch.zeros((3,4),dtype=torch.float64),d.Config(game_iterations=8))
            self.assertTrue(bool(allowed[torch.arange(3),a].all()))

    def test_chunk_size_does_not_change_experiment(self):
        cfg=d.Config(preset='smoke',budget=12,eig_samples=4,game_iterations=8,device='cpu')
        cases=[c for c in d.build_cases(cfg) if c['domain']=='kinetics' and c['regime']=='full'][:2]
        for method in ('eig_safe','deficit_safe'):
            together,_=d.run_batch(cases,method,cfg)
            apart=[d.run_batch([c],method,cfg)[0][0] for c in cases]
            for x,y in zip(together,apart):
                for k in ['decision','answer','cost','queries','correct','surviving_atoms']:
                    self.assertEqual(x[k],y[k])

    def test_resume_and_configuration_guard(self):
        cfg=d.Config(preset='smoke',budget=2,eig_samples=2,game_iterations=2,
                     methods=('random_safe',),device='cpu',max_minutes=0)
        with tempfile.TemporaryDirectory() as td:
            d.run_suite(cfg,td); first=d.load_rows(td)
            d.run_suite(dataclasses.replace(cfg,batch=1),td)
            self.assertEqual(first,d.load_rows(td))
            with self.assertRaises(ValueError): d.run_suite(dataclasses.replace(cfg,alpha=.1),td)

    def test_guarantee_flag_does_not_extend_to_stress(self):
        cfg=d.Config(preset='pilot',budget=2,eig_samples=2,game_iterations=2,device='cpu')
        for scenario in ('closed','off_grid','missing_family','noise_mismatch'):
            c=next(c for c in d.build_cases(cfg) if c['scenario']==scenario and c['regime']=='full')
            row=d.run_batch([c],'random_safe',cfg)[0][0]
            self.assertEqual(row['guarantee_applicable'],scenario=='closed')
            if scenario in ('off_grid','missing_family'): self.assertFalse(row['true_mean_in_catalog'])

    def test_error_coverage_smoke_under_adaptive_sampling(self):
        # Regression gate, not a proof and not a publication-level calibration study.
        cfg=d.Config(preset='smoke',budget=24,eig_samples=4,game_iterations=8,device='cpu',trace_per_group=0)
        template=next(c for c in d.build_cases(cfg) if c['domain']=='kinetics' and c['regime']=='full')
        cases=[]
        for rep in range(200):
            c=dict(template,rep=rep); c['id']=d.digest([template['id'],'coverage-test',rep])[:24]; cases.append(c)
        rows,_=d.run_batch(cases,'deficit_safe',cfg)
        errors=sum(r['false_declaration'] for r in rows)
        self.assertLessEqual(errors,20) # conservative regression alarm, alpha=.05

if __name__=='__main__': unittest.main(verbosity=2)
