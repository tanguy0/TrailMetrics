/* @ds-bundle: {"format":4,"namespace":"TAGG","components":[{"name":"Button"},{"name":"Chip"},{"name":"KpiTile"},{"name":"Field"},{"name":"Toggle"},{"name":"NavRail"},{"name":"PageHeader"},{"name":"Panel"},{"name":"PlotCard"},{"name":"DataTable"},{"name":"SessionCard"},{"name":"Teaser"},{"name":"AccessGrid"}]} */
(function(){
  var R = window.React, h = R.createElement;
  function cx(){ return Array.prototype.slice.call(arguments).filter(Boolean).join(" "); }
  function Button(p){ var v=p.variant||"primary", s=p.size||"md";
    return h(p.href?"a":"button", Object.assign({}, p, {variant:undefined,size:undefined,wide:undefined,icon:undefined,
      className:cx("tm-btn", v!=="primary"&&"tm-btn--"+v, s==="sm"&&"tm-btn--sm", p.wide&&"tm-btn--wide", p.className)}), p.icon, p.children); }
  function Chip(p){ return h("span",{className:cx("tm-chip", p.tone&&p.tone!=="neutral"&&"tm-chip--"+p.tone, p.dot&&"tm-chip--dot", p.className)}, p.children); }
  function KpiTile(p){ return h("div",{className:"tm-kpi"},
    h("div",{className:"tm-kpi__label"},p.label),
    h("div",{className:"tm-kpi__value"}, h("span",{className:"tm-kpi__num"},p.value), p.unit&&h("span",{className:"tm-kpi__unit"},p.unit)),
    p.delta&&h("div",{className:cx("tm-kpi__delta", p.trend&&"tm-kpi__delta--"+p.trend)},p.delta)); }
  function Field(p){ var id=p.id||("f"+Math.random().toString(36).slice(2,7));
    return h("label",{className:"tm-field",htmlFor:id}, p.label,
      p.options ? h("select",{id:id,className:"tm-select",value:p.value,onChange:p.onChange}, p.options.map(function(o){return h("option",{key:o.value||o,value:o.value||o},o.label||o);}))
                : h("input",{id:id,className:"tm-input",type:p.type||"text",value:p.value,placeholder:p.placeholder,onChange:p.onChange})); }
  function Toggle(p){ return h("label",{className:"tm-toggle"}, h("input",{type:"checkbox",checked:!!p.checked,onChange:p.onChange}), h("span",{className:"tm-toggle__track"}), p.children); }
  function NavRail(p){ return h("nav",{className:"tm-rail","aria-label":"Navigation"},
    h("a",{className:"tm-rail__brand",href:p.homeHref||"/"}, p.brand),
    p.switcher && h("button",{type:"button",className:"tm-rail__switcher"}, p.switcher),
    h("ul",{className:"tm-rail__nav"}, (p.items||[]).map(function(it){ return h("li",{key:it.href},
      h("a",{href:it.disabled?undefined:it.href,className:cx("tm-rail__link",it.active&&"tm-rail__link--active",it.disabled&&"tm-rail__link--disabled"),"aria-current":it.active?"page":undefined,"aria-disabled":it.disabled||undefined}, it.icon, it.label)); })),
    p.footer && h("div",{className:"tm-rail__footer"}, p.footer)); }
  function PageHeader(p){ return h("header",{className:"tm-page-header"},
    h("div",null, p.kicker&&h("div",{className:"tm-page-header__kicker"},p.kicker), h("h1",{className:"tm-page-header__title"},p.title), p.subtitle&&h("div",{className:"tm-page-header__sub"},p.subtitle)),
    p.actions&&h("div",{className:"tm-page-header__actions"},p.actions)); }
  function Panel(p){ return h("section",{className:"tm-panel"},
    h("div",{className:"tm-panel__head"},
      h("div",{className:"tm-panel__lead"}, p.index&&h("div",{className:"tm-panel__index"},p.index),
        h("div",null, h("h2",{className:"tm-panel__title"},p.title), p.meta&&h("div",{className:"tm-panel__meta"},p.meta), p.description&&h("p",{className:"tm-panel__desc"},p.description))),
      p.actions&&h("div",{className:"tm-panel__actions"},p.actions)),
    h("div",{className:"tm-plot-grid"}, p.children)); }
  function PlotCard(p){ return h("div",{className:cx("tm-plot", p.span&&"tm-plot--"+p.span)},
    h("div",{className:"tm-plot__head"}, h("div",null, h("div",{className:"tm-plot__title"},p.title), p.subtitle&&h("div",{className:"tm-plot__sub"},p.subtitle)), p.tag&&h(Chip,null,p.tag)),
    p.children, p.legend&&h("div",{className:"tm-plot__legend"},p.legend)); }
  function DataTable(p){ return h("table",{className:"tm-table"},
    h("thead",null,h("tr",null,(p.columns||[]).map(function(c){return h("th",{key:c.key,className:cx(c.numeric&&"is-num")},c.label);}))),
    h("tbody",null,(p.rows||[]).map(function(r,i){return h("tr",{key:r.id||i,className:cx(r.best&&"is-best")},(p.columns||[]).map(function(c){return h("td",{key:c.key,className:cx(c.numeric&&"is-num",c.date&&"is-date")},r[c.key]);}));}))); }
  function SessionCard(p){
    if(p.kind==="goal") return h("div",{className:cx("tm-session tm-session--goal",p.secondary&&"is-secondary")}, p.icon, p.title);
    if(p.kind==="planned") return h("div",{className:"tm-session tm-session--planned"}, h("div",{className:"tm-session__title"},p.icon,p.title), p.body&&h("div",{className:"tm-session__body"},p.body));
    return h("div",{className:"tm-session","data-sport":p.sport||"other"},
      h("div",{className:"tm-session__title"},p.icon,h("span",null,p.title)),
      p.stats&&h("div",{className:"tm-session__stats"}, p.stats.map(function(s,i){return h(i===0?"b":"span",{key:i},s);})),
      p.tags&&h("div",{className:"tm-session__tags"},p.tags)); }

  function Teaser(p){ return h("section",{className:"tm-teaser","aria-label":p.title},
    h("div",{className:"tm-teaser__bg","aria-hidden":"true"}, p.background),
    h("div",{className:"tm-teaser__card"},
      h("div",{className:"tm-teaser__kicker"},p.kicker||"Gratuit · avec Strava"),
      h("h2",{className:"tm-teaser__title"},p.title),
      p.bullets&&h("ul",{className:"tm-teaser__list"},p.bullets.map(function(b,i){return h("li",{key:i},b);})),
      h("div",{className:"tm-teaser__actions"},p.actions),
      p.fine&&h("div",{className:"tm-teaser__fine"},p.fine))); }
  function AccessGrid(p){ function col(c,locked){ return h("div",{className:"tm-access__col"},
      h("div",{className:"tm-access__head"},c.title, c.chip),
      (c.items||[]).map(function(it){ return h("a",{key:it.href,href:it.href,className:cx("tm-access__item",locked&&"tm-access__item--locked")}, it.icon,
        h("div",null,h("div",{className:"tm-access__title"},it.title),h("div",{className:"tm-access__desc"},it.desc))); })); }
    return h("div",{className:"tm-access"}, col(p.open,false), col(p.locked,true)); }
  window.TAGG = window.TAGG || {};
  Object.assign(window.TAGG, {Button:Button,Chip:Chip,KpiTile:KpiTile,Field:Field,Toggle:Toggle,NavRail:NavRail,PageHeader:PageHeader,Panel:Panel,PlotCard:PlotCard,DataTable:DataTable,SessionCard:SessionCard,Teaser:Teaser,AccessGrid:AccessGrid});
})();
